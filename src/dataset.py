import os
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Dict

import torch  # pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Dataset, random_split  # pyrefly: ignore [missing-import]
import torchvision.transforms as transforms  # pyrefly: ignore [missing-import]
from torchvision.datasets import ImageFolder  # pyrefly: ignore [missing-import]
from PIL import Image  # pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]

# Standard ImageNet Mean and Standard Deviation for Transfer Learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns training and validation/testing data augmentation pipelines.
    
    Training pipeline includes field-robust transformations:
    - RandomResizedCrop: Simulates varying photo distances & zoom
    - RandomHorizontalFlip / RandomVerticalFlip: Simulates arbitrary leaf orientation
    - RandomRotation: Simulates camera tilt
    - ColorJitter: Simulates variable field lighting (sunlight vs shadow)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_test_transform

class TransformedDataset(Dataset):
    """
    Wrapper Dataset to apply specific transforms to a Subset of ImageFolder dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if isinstance(x, Image.Image):
            x = x.convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[int, str], torch.Tensor]:
    """
    Creates train, validation, and test PyTorch DataLoaders from PlantVillage directory.
    
    Returns:
        (train_loader, val_loader, test_loader, class_names, idx_to_class, class_weights)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist. Run download_data.py first.")

    train_tf, val_tf = get_transforms(img_size=img_size)

    # Load raw dataset without transforms first so we can apply split-specific transforms
    raw_dataset = ImageFolder(root=str(data_path), transform=None)
    class_names = raw_dataset.classes
    idx_to_class = {v: k for k, v in raw_dataset.class_to_idx.items()}

    total_count = len(raw_dataset)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_count, val_count, test_count], generator=generator
    )

    # Wrap subsets with respective transform wrappers
    train_dataset = TransformedDataset(train_subset, transform=train_tf)
    val_dataset = TransformedDataset(val_subset, transform=val_tf)
    test_dataset = TransformedDataset(test_subset, transform=val_tf)

    # Check CUDA memory pinning hardware availability
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )

    # Compute inverse-frequency class weights for balanced loss
    label_counts = Counter(raw_dataset.targets)
    num_samples = len(raw_dataset)
    num_classes = len(class_names)
    class_weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        count = label_counts.get(cls_idx, 1)
        class_weights[cls_idx] = num_samples / (num_classes * count)

    print(f"DataLoaders Created Successfully:")
    print(f"  - Classes ({len(class_names)}): {class_names[:5]}...")
    print(f"  - Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  - Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"  - Test samples:  {len(test_dataset)} ({len(test_loader)} batches)")
    print(f"  - Class weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

    return train_loader, val_loader, test_loader, class_names, idx_to_class, class_weights

def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalization for plotting PyTorch tensors as images."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        train_ld, val_ld, test_ld, classes, idx_map, weights = create_dataloaders(args.data_dir, batch_size=16)
        batch_imgs, batch_lbls = next(iter(train_ld))
        print(f"Sample Batch Image Tensor Shape: {batch_imgs.shape}")
        print(f"Sample Batch Label Tensor Shape: {batch_lbls.shape}")
