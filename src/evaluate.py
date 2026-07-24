import os
import argparse
from pathlib import Path

import torch  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]
import seaborn as sns  # pyrefly: ignore [missing-import]
from sklearn.metrics import classification_report, confusion_matrix  # pyrefly: ignore [missing-import]

try:
    from .dataset import create_dataloaders, denormalize_image
    from .models import load_model_from_checkpoint
    from .calibration import calibrate_temperature
except ImportError:
    from dataset import create_dataloaders, denormalize_image
    from models import load_model_from_checkpoint
    from calibration import calibrate_temperature

def evaluate_model(checkpoint_path: str, data_dir: str = "data/PlantVillage", batch_size: int = 32, reports_dir: str = "reports"):
    """
    Evaluates saved model on the test dataset split and generates diagnostic charts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load model (handles flat + hierarchical architectures)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    model_type = checkpoint.get("model_type", "resnet18")
    class_names = checkpoint["class_names"]
    
    # 1. Create Test and Validation Loaders (val needed for calibration)
    _, val_loader, test_loader, _, _, _ = create_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # 3. Collect Predictions
    all_preds = []
    all_targets = []
    all_imgs = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
            if len(all_imgs) < 16:
                for img in images.cpu():
                    all_imgs.append(img)

    # 4. Compute Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = np.mean(all_preds == all_targets) * 100.0
    print(f"\n" + "="*50)
    print(f" TEST SET EVALUATION RESULTS ({model_type}) ")
    print("="*50)
    print(f"Overall Test Accuracy: {acc:.2f}%")
    print("-" * 50)
    
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # 5. Generate & Save Confusion Matrix
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_type} (Test Accuracy: {acc:.2f}%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    cm_file = out_dir / f"confusion_matrix_{model_type}.png"
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix Plot -> {cm_file}")

    # 6. Plot Sample Predictions Grid
    plt.figure(figsize=(12, 12))
    for i in range(min(16, len(all_imgs))):
        plt.subplot(4, 4, i + 1)
        img = denormalize_image(all_imgs[i]).numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        
        true_label = class_names[all_targets[i]]
        pred_label = class_names[all_preds[i]]
        
        color = "green" if true_label == pred_label else "red"
        plt.imshow(img)
        plt.title(f"True: {true_label.split('___')[-1]}\nPred: {pred_label.split('___')[-1]}", color=color, fontsize=8)
        plt.axis("off")
        
    plt.tight_layout()
    samples_file = out_dir / f"sample_predictions_{model_type}.png"
    plt.savefig(samples_file, dpi=300)
    plt.close()
    print(f"Saved Sample Predictions Grid -> {samples_file}\n")

    # 7. Temperature Calibration (saves T into checkpoint for calibrated inference)
    print("Running temperature calibration on validation set...")
    temperature = calibrate_temperature(model, val_loader, device)
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)
    print(f"Calibrated temperature T={temperature:.4f} saved to checkpoint.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_dir)
