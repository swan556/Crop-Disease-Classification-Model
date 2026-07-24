"""
Temperature Scaling for post-training model calibration.

After training, a model's softmax probabilities often don't reflect true confidence levels
(i.e., a prediction of 95% confidence doesn't mean it's correct 95% of the time).

Temperature scaling learns a single scalar T on the validation set:
    calibrated_probs = softmax(logits / T)

When T > 1: softens predictions (reduces overconfidence)
When T < 1: sharpens predictions
When T = 1: no change (uncalibrated)

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import argparse

import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]
from torch.optim import LBFGS  # pyrefly: ignore [missing-import]


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter T for calibrating model logits."""

    def __init__(self):
        super().__init__()
        # Initialize T=1.0 (uncalibrated)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature


def calibrate_temperature(model, val_loader, device, max_iter=50):
    """
    Learns the optimal temperature T on the validation set using LBFGS.

    This collects all validation logits, then optimizes T to minimize
    the negative log-likelihood (CrossEntropyLoss) on the validation set.

    Args:
        model: trained model in eval mode
        val_loader: validation DataLoader
        device: torch.device
        max_iter: LBFGS max iterations

    Returns:
        optimal temperature value (float)
    """
    model.eval()

    # Collect all logits and labels from validation set
    all_logits = []
    all_labels = []

    print("  Collecting validation logits for calibration...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    # Optimize temperature using LBFGS
    temp_scaler = TemperatureScaler().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = LBFGS([temp_scaler.temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_scaler(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temp = temp_scaler.temperature.item()
    print(f"  Calibration complete. Optimal temperature T = {optimal_temp:.4f}")

    return optimal_temp


def calibrate_checkpoint(checkpoint_path, data_dir="data/PlantVillage", batch_size=32):
    """
    Calibrates a saved model checkpoint and saves the temperature value back.

    Usage:
        python src/calibration.py --checkpoint checkpoints/best_model_resnet18.pth
    """
    try:
        from .dataset import create_dataloaders
        from .models import load_model_from_checkpoint
    except ImportError:
        from dataset import create_dataloaders
        from models import load_model_from_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f" TEMPERATURE CALIBRATION")
    print(f"{'='*50}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    # Load model
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    # Get validation loader
    _, val_loader, _, _, _, _ = create_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # Calibrate
    temperature = calibrate_temperature(model, val_loader, device)

    # Save back to checkpoint
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)
    print(f"  Temperature T={temperature:.4f} saved to {checkpoint_path}")
    print(f"{'='*50}\n")

    return temperature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate model temperature")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    args = parser.parse_args()

    calibrate_checkpoint(args.checkpoint, args.data_dir)
