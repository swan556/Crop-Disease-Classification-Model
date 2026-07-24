"""
Hierarchical classification utilities for Crop Disease Classification.

The 38 PlantVillage classes encode a natural hierarchy:
    Plant___Disease (e.g., "Apple___Apple_scab", "Tomato___healthy")

This module parses that hierarchy and provides:
- Mappings between class indices, plant species, and disease conditions
- HierarchicalLoss: combined plant + disease loss for two-head training
"""

import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]
from typing import List, Dict


def build_hierarchy(class_names: List[str]) -> Dict:
    """
    Parses class names into a plant/disease hierarchy.

    The PlantVillage dataset uses "Plant___Disease" naming convention.
    This function extracts the plant species from each class name and builds
    bidirectional mappings between class indices and plant species indices.

    Args:
        class_names: List of class names in format "Plant___Disease"

    Returns:
        Dictionary containing:
        - plant_names: sorted list of unique plant species
        - num_plants: number of unique plant species
        - class_to_plant_idx: tensor mapping class_idx -> plant_idx
        - plant_to_classes: dict mapping plant_idx -> list of class_idx
        - plant_to_idx: dict mapping plant_name -> plant_idx
    """
    plant_set = set()
    class_to_plant = {}

    for idx, name in enumerate(class_names):
        parts = name.split("___")
        plant = parts[0] if len(parts) >= 2 else name
        class_to_plant[idx] = plant
        plant_set.add(plant)

    plant_names = sorted(plant_set)
    plant_to_idx = {name: idx for idx, name in enumerate(plant_names)}

    # Build class_idx -> plant_idx mapping tensor
    class_to_plant_idx = torch.zeros(len(class_names), dtype=torch.long)
    for cls_idx, plant_name in class_to_plant.items():
        class_to_plant_idx[cls_idx] = plant_to_idx[plant_name]

    # Build plant_idx -> [class_idx, ...] mapping
    plant_to_classes = {}
    for cls_idx, plant_name in class_to_plant.items():
        pidx = plant_to_idx[plant_name]
        if pidx not in plant_to_classes:
            plant_to_classes[pidx] = []
        plant_to_classes[pidx].append(cls_idx)

    print(f"  Hierarchy built: {len(plant_names)} plant species from {len(class_names)} disease classes")
    print(f"  Plants: {plant_names}")

    return {
        "plant_names": plant_names,
        "num_plants": len(plant_names),
        "class_to_plant_idx": class_to_plant_idx,
        "plant_to_classes": plant_to_classes,
        "plant_to_idx": plant_to_idx,
    }


class HierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical two-head classification.

    L_total = L_disease + lambda * L_plant

    Where:
    - L_disease: CrossEntropyLoss on the full 38-class disease prediction (primary task)
    - L_plant: CrossEntropyLoss on the 14-class plant species prediction (auxiliary task)
    - lambda: weight for the plant loss (default: 0.5)

    The plant head acts as a regularizer, forcing the backbone to learn
    plant-discriminative features that prevent cross-species confusions
    (e.g., predicting a Corn disease on an Apple leaf).
    """
    def __init__(self, class_to_plant_idx: torch.Tensor,
                 disease_criterion: nn.Module = None,
                 plant_lambda: float = 0.5):
        super().__init__()
        self.register_buffer("class_to_plant_idx", class_to_plant_idx)
        self.disease_criterion = disease_criterion or nn.CrossEntropyLoss()
        self.plant_criterion = nn.CrossEntropyLoss()
        self.plant_lambda = plant_lambda

    def forward(self, plant_logits: torch.Tensor, disease_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Computes combined hierarchical loss.

        Args:
            plant_logits: (B, num_plants) logits from plant classification head
            disease_logits: (B, num_classes) logits from disease classification head
            targets: (B,) ground-truth class indices (0-37)

        Returns:
            Combined scalar loss
        """
        # Primary: disease classification loss
        disease_loss = self.disease_criterion(disease_logits, targets)

        # Auxiliary: plant species loss (targets converted via hierarchy mapping)
        plant_targets = self.class_to_plant_idx[targets]
        plant_loss = self.plant_criterion(plant_logits, plant_targets)

        return disease_loss + self.plant_lambda * plant_loss


if __name__ == "__main__":
    # Smoke test with sample class names
    sample_classes = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy",
        "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
        "Tomato___Early_blight", "Tomato___healthy"
    ]

    info = build_hierarchy(sample_classes)
    print(f"\nClass-to-Plant mapping: {info['class_to_plant_idx']}")
    print(f"Plant-to-Classes: {info['plant_to_classes']}")

    # Test loss computation
    loss_fn = HierarchicalLoss(info["class_to_plant_idx"])
    plant_logits = torch.randn(4, info["num_plants"])
    disease_logits = torch.randn(4, len(sample_classes))
    targets = torch.tensor([0, 2, 4, 6])

    loss = loss_fn(plant_logits, disease_logits, targets)
    print(f"Hierarchical Loss: {loss.item():.4f}")
