import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]
import torch.optim as optim  # pyrefly: ignore [missing-import]
from torch.optim.lr_scheduler import CosineAnnealingLR  # pyrefly: ignore [missing-import]
from torch.amp import autocast, GradScaler  # pyrefly: ignore [missing-import]
from tqdm import tqdm  # pyrefly: ignore [missing-import]

try:
    from .dataset import create_dataloaders
    from .models import create_model, create_hierarchical_model, HierarchicalClassifier
    from .hierarchy import build_hierarchy, HierarchicalLoss
except ImportError:
    from dataset import create_dataloaders
    from models import create_model, create_hierarchical_model, HierarchicalClassifier
    from hierarchy import build_hierarchy, HierarchicalLoss


def cutmix_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """
    Apply CutMix augmentation to a batch of images.
    
    Cuts a random rectangular patch from a shuffled copy of the batch and
    pastes it onto the original images. Labels are mixed proportionally
    to the area ratio, forcing the model to learn from partial views.
    
    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong 
    Classifiers with Localizable Features" (ICCV 2019)
    
    Args:
        images: (B, C, H, W) batch of images
        labels: (B,) batch of labels
        alpha: Beta distribution parameter (higher = more uniform mixing)
    
    Returns:
        (mixed_images, labels_a, labels_b, lam) where lam is the mixing ratio
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    
    _, _, H, W = images.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cy = random.randint(0, H - 1)
    cx = random.randint(0, W - 1)
    
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    
    images_mixed = images.clone()
    images_mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual cropped area
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(H * W)
    
    return images_mixed, labels, labels[index], lam


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    cutmix_prob: float = 0.5,
    hierarchical_loss_fn: nn.Module = None
) -> Tuple[float, float]:
    """
    Executes one single epoch of training.
    
    Supports:
    - AMP mixed precision via scaler
    - CutMix augmentation (applied with probability cutmix_prob)
    - Hierarchical two-head models via hierarchical_loss_fn
    - Gradient clipping (max_norm=1.0)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and device.type == "cuda"
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # CutMix augmentation (stochastic)
        apply_cutmix = cutmix_prob > 0 and random.random() < cutmix_prob
        if apply_cutmix:
            images, labels_a, labels_b, lam = cutmix_data(images, labels)
        
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            
            if apply_cutmix:
                if hierarchical_loss_fn is not None and isinstance(outputs, tuple):
                    plant_logits, disease_logits = outputs
                    loss = (lam * hierarchical_loss_fn(plant_logits, disease_logits, labels_a) +
                            (1 - lam) * hierarchical_loss_fn(plant_logits, disease_logits, labels_b))
                    outputs_for_acc = disease_logits
                else:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    outputs_for_acc = outputs
            else:
                if hierarchical_loss_fn is not None and isinstance(outputs, tuple):
                    plant_logits, disease_logits = outputs
                    loss = hierarchical_loss_fn(plant_logits, disease_logits, labels)
                    outputs_for_acc = disease_logits
                else:
                    loss = criterion(outputs, labels)
                    outputs_for_acc = outputs
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs_for_acc, 1)
        if apply_cutmix:
            # Approximate accuracy for CutMix-blended batches
            correct += (lam * (preds == labels_a).sum().item() +
                       (1 - lam) * (preds == labels_b).sum().item())
        else:
            correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        current_acc = (correct / total) * 100.0
        current_loss = running_loss / total
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates model performance on validation/testing dataset.
    
    Steps:
    1. Set model to evaluation mode (model.eval())
    2. Disable gradient tracking (with torch.no_grad():) to conserve VRAM & speed up computation
    3. Compute predictions and calculate validation loss and accuracy
    
    Note: Hierarchical models return only disease logits in eval mode,
    so this function works identically for both flat and hierarchical models.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def run_training(
    model_type: str = "resnet18",
    data_dir: str = "data/PlantVillage",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
    patience: int = 7,
    hierarchical: bool = True,
    cutmix_prob: float = 0.5
):
    """
    Complete Training Pipeline Execution Engine.

    Key features:
    - Discriminative learning rates (backbone lr/10, head lr)
    - CosineAnnealingLR for smooth convergence
    - Class-weighted CrossEntropyLoss with label smoothing for imbalanced data
    - AMP mixed precision training on CUDA
    - Gradient clipping (max_norm=1.0)
    - CutMix augmentation for regularization
    - Hierarchical two-head classification (plant species + disease)
    - Early stopping with configurable patience
    """
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*60)
    print(f" STARTING TRAINING PIPELINE ")
    print("="*60)
    print(f"Device Acceleration: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model Architecture:  {model_type}")
    print(f"Hierarchical Mode:   {hierarchical}")
    print(f"Epochs:              {epochs}")
    print(f"Batch Size:          {batch_size}")
    print(f"Initial Learning Rate: {lr} (backbone: {lr/10:.1e})")
    print(f"CutMix Probability:  {cutmix_prob}")
    print(f"Early Stopping:      patience={patience}")
    print("="*60 + "\n")

    # 1. Create DataLoaders (returns class_weights)
    train_loader, val_loader, test_loader, class_names, idx_to_class, class_weights = create_dataloaders(
        data_dir=data_dir, batch_size=batch_size, seed=seed
    )

    num_classes = len(class_names)
    class_weights = class_weights.to(device)

    # 2. Build model (flat or hierarchical)
    hierarchy_info = None
    hierarchical_loss_fn = None

    if hierarchical:
        hierarchy_info = build_hierarchy(class_names)
        num_plants = hierarchy_info["num_plants"]

        model = create_hierarchical_model(model_type, num_classes, num_plants)
        model = model.to(device)

        # Disease criterion with class weights and label smoothing
        disease_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # Hierarchical loss: L_disease + 0.5 * L_plant
        hierarchical_loss_fn = HierarchicalLoss(
            class_to_plant_idx=hierarchy_info["class_to_plant_idx"],
            disease_criterion=disease_criterion,
            plant_lambda=0.5
        ).to(device)

        # Flat criterion for validation (model returns disease logits only in eval mode)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        model = create_model(model_type=model_type, num_classes=num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 3. Optimizer with discriminative learning rates
    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(head_lr=lr, backbone_lr=lr / 10)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        print(f"  Using discriminative LR: backbone={lr/10:.1e}, head={lr:.1e}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Cosine Annealing LR Scheduler for smooth convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. AMP GradScaler for mixed precision (CUDA only)
    scaler = GradScaler() if device.type == "cuda" else None
    if scaler:
        print("  AMP Mixed Precision: Enabled")

    # 6. Checkpoint Management Setup
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    best_model_file = ckpt_path / f"best_model_{model_type}.pth"

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"Epoch [{epoch}/{epochs}]")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            cutmix_prob=cutmix_prob,
            hierarchical_loss_fn=hierarchical_loss_fn
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step LR Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | LR: {current_lr:.6f}")

        # Checkpointing logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            checkpoint_data = {
                "epoch": epoch,
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "idx_to_class": idx_to_class,
                "hierarchical": hierarchical,
            }
            if hierarchical and hierarchy_info:
                checkpoint_data["num_plants"] = hierarchy_info["num_plants"]
                checkpoint_data["plant_names"] = hierarchy_info["plant_names"]
            
            torch.save(checkpoint_data, best_model_file)
            print(f"  -> Best model saved! (Val Acc: {best_val_acc:.2f}%) -> {best_model_file.name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered after {patience} epochs with no improvement.")
                break
            
        print("-" * 60)

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes!")
    print(f"Best Validation Accuracy Achieved: {best_val_acc:.2f}%")
    print(f"Model Checkpoint saved at: {best_model_file.resolve()}\n")

    return model, history, best_model_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crop Leaf Disease Model")
    parser.add_argument("--model_type", type=str, default="resnet18",
                        choices=["custom_cnn", "resnet18", "resnet50", "efficientnet_b2"])
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--no-hierarchical", action="store_true", help="Disable hierarchical classification")
    parser.add_argument("--cutmix_prob", type=float, default=0.5, help="CutMix probability (0 to disable)")
    args = parser.parse_args()

    run_training(
        model_type=args.model_type,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        hierarchical=not args.no_hierarchical,
        cutmix_prob=args.cutmix_prob
    )
