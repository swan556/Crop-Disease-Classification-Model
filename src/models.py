import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]
import torchvision.models as models  # pyrefly: ignore [missing-import]

class LeafDiseaseCNN(nn.Module):
    """
    Custom Convolutional Neural Network built from scratch for Crop Leaf Disease Diagnosis.
    
    Architecture:
    - 4 Convolutional Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D)
    - Adaptive Global Average Pooling
    - Fully Connected Classifier Head with Dropout regularization
    """
    def __init__(self, num_classes: int = 38, in_channels: int = 3):
        super(LeafDiseaseCNN, self).__init__()
        
        # Block 1: Input (3, 224, 224) -> Output (32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Input (32, 112, 112) -> Output (64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Input (64, 56, 56) -> Output (128, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: Input (128, 28, 28) -> Output (256, 14, 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive Pooling collapses spatial dimensions to (1, 1) regardless of input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension after global pooling (used by HierarchicalClassifier)
        self.feature_dim = 256
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class ResNetLeafClassifier(nn.Module):
    """
    Fine-tuned ResNet Architecture (ResNet18 / ResNet50) for Crop Leaf Disease Diagnosis.
    
    Supports:
    - Pre-trained ImageNet weights initialization
    - Optional backbone parameter freezing (Feature Extractor vs Fine-tuning)
    - Custom replacement of final linear classifier layer
    """
    def __init__(self, num_classes: int = 38, model_name: str = "resnet18", freeze_backbone: bool = False):
        super(ResNetLeafClassifier, self).__init__()
        
        if model_name.lower() == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
        elif model_name.lower() == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {model_name}. Choose 'resnet18' or 'resnet50'.")

        # Store raw feature dimension (used by HierarchicalClassifier)
        self.feature_dim = in_features

        if freeze_backbone:
            print("Freezing ResNet backbone layers (Feature Extraction mode)...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head with a deeper bottleneck
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_param_groups(self, head_lr: float, backbone_lr: float):
        """
        Returns parameter groups with discriminative learning rates.
        The pretrained backbone uses a lower LR to preserve ImageNet features,
        while the randomly-initialized head uses a higher LR to learn quickly.
        """
        head_params = list(self.backbone.fc.parameters())
        head_param_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_param_ids]

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr}
        ]


class EfficientNetLeafClassifier(nn.Module):
    """
    Fine-tuned EfficientNet-B2 for Crop Leaf Disease Diagnosis.
    
    EfficientNet uses depthwise separable convolutions and compound scaling,
    offering better feature abstraction than ResNet with similar compute.
    Significantly reduces texture bias compared to standard CNNs.
    
    Parameters: ~9M (vs ResNet18's ~11M, ResNet50's ~25M)
    """
    def __init__(self, num_classes: int = 38, freeze_backbone: bool = False):
        super(EfficientNetLeafClassifier, self).__init__()
        
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)
        
        # EfficientNet-B2 feature dimension: 1408
        in_features = self.backbone.classifier[1].in_features
        self.feature_dim = in_features
        
        if freeze_backbone:
            print("Freezing EfficientNet backbone layers (Feature Extraction mode)...")
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classification head with bottleneck
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_param_groups(self, head_lr: float, backbone_lr: float):
        """Returns parameter groups with discriminative learning rates."""
        head_params = list(self.backbone.classifier.parameters())
        head_param_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_param_ids]
        
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr}
        ]


class HierarchicalClassifier(nn.Module):
    """
    Two-head hierarchical classifier wrapper.
    
    Wraps any backbone (ResNet, EfficientNet, Custom CNN) and replaces its
    single classification head with two heads:
    - Plant Head: predicts plant species (14 classes) — auxiliary regularizer
    - Disease Head: predicts disease condition (38 classes) — primary task
    
    During training: returns (plant_logits, disease_logits) for HierarchicalLoss
    During evaluation: returns only disease_logits for compatibility with inference
    """
    def __init__(self, feature_extractor: nn.Module, feature_dim: int,
                 num_plants: int, num_classes: int):
        super(HierarchicalClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        
        # Plant species classification head (auxiliary)
        self.plant_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_plants)
        )
        
        # Disease classification head (primary)
        self.disease_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        plant_logits = self.plant_head(features)
        disease_logits = self.disease_head(features)
        
        if self.training:
            return plant_logits, disease_logits
        # In eval mode, return only disease logits for inference compatibility
        return disease_logits
    
    def get_param_groups(self, head_lr: float, backbone_lr: float):
        """Returns discriminative LR groups: backbone (low LR) + both heads (high LR)."""
        head_params = list(self.plant_head.parameters()) + list(self.disease_head.parameters())
        backbone_params = list(self.feature_extractor.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr}
        ]


def create_model(model_type: str = "resnet18", num_classes: int = 38, freeze_backbone: bool = False) -> nn.Module:
    """Factory function to build selected model architecture."""
    if model_type.lower() == "custom_cnn":
        model = LeafDiseaseCNN(num_classes=num_classes)
    elif model_type.lower() in ("resnet18", "resnet50"):
        model = ResNetLeafClassifier(num_classes=num_classes, model_name=model_type, freeze_backbone=freeze_backbone)
    elif model_type.lower() == "efficientnet_b2":
        model = EfficientNetLeafClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: ['custom_cnn', 'resnet18', 'resnet50', 'efficientnet_b2']")
    return model


def create_hierarchical_model(model_type: str, num_classes: int, num_plants: int,
                               freeze_backbone: bool = False) -> HierarchicalClassifier:
    """
    Creates a hierarchical two-head model from a base architecture.
    
    The base model's classifier head is replaced with nn.Identity() to extract
    raw features, then two new heads (plant + disease) are attached.
    """
    base_model = create_model(model_type, num_classes, freeze_backbone)
    
    # Extract feature dimension and replace classifier with Identity to get raw features
    if isinstance(base_model, ResNetLeafClassifier):
        feature_dim = base_model.feature_dim
        base_model.backbone.fc = nn.Identity()
    elif isinstance(base_model, EfficientNetLeafClassifier):
        feature_dim = base_model.feature_dim
        base_model.backbone.classifier = nn.Identity()
    elif isinstance(base_model, LeafDiseaseCNN):
        feature_dim = base_model.feature_dim
        # Keep only Flatten, remove classification layers
        base_model.classifier = nn.Sequential(nn.Flatten())
    else:
        raise ValueError(f"Unsupported model type for hierarchical: {model_type}")
    
    return HierarchicalClassifier(base_model, feature_dim, num_plants, num_classes)


def load_model_from_checkpoint(checkpoint_path: str, device=None):
    """
    Load model from checkpoint, handling flat, hierarchical, and all backbone types.
    
    This is the unified model loading function used by predict.py, evaluate.py,
    and calibration.py. It reads the checkpoint metadata to determine which
    architecture to instantiate.
    
    Returns:
        (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "resnet18")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    
    if checkpoint.get("hierarchical", False):
        num_plants = checkpoint["num_plants"]
        model = create_hierarchical_model(model_type, num_classes, num_plants)
    else:
        model = create_model(model_type=model_type, num_classes=num_classes)
    
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test Custom CNN
    cnn = create_model("custom_cnn", num_classes=38)
    out_cnn = cnn(dummy_input)
    print(f"Custom CNN Output Shape: {out_cnn.shape}")
    
    # Test ResNet18
    resnet = create_model("resnet18", num_classes=38)
    out_res = resnet(dummy_input)
    print(f"ResNet18 Output Shape: {out_res.shape}")
    
    # Test EfficientNet-B2
    effnet = create_model("efficientnet_b2", num_classes=38)
    out_eff = effnet(dummy_input)
    print(f"EfficientNet-B2 Output Shape: {out_eff.shape}")
    
    # Test Hierarchical Model (train mode returns tuple, eval mode returns disease logits)
    hier = create_hierarchical_model("resnet18", num_classes=38, num_plants=14)
    hier.train()
    plant_out, disease_out = hier(dummy_input)
    print(f"Hierarchical [train] - Plant: {plant_out.shape}, Disease: {disease_out.shape}")
    
    hier.eval()
    eval_out = hier(dummy_input)
    print(f"Hierarchical [eval]  - Output: {eval_out.shape}")
