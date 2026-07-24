import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch  # pyrefly: ignore [missing-import]
import torch.nn.functional as F  # pyrefly: ignore [missing-import]
from PIL import Image  # pyrefly: ignore [missing-import]
import torchvision.transforms as transforms  # pyrefly: ignore [missing-import]

try:
    from .dataset import IMAGENET_MEAN, IMAGENET_STD
    from .models import load_model_from_checkpoint
except ImportError:
    from dataset import IMAGENET_MEAN, IMAGENET_STD
    from models import load_model_from_checkpoint

# Agricultural guidance dictionary for farming advice
TREATMENT_ADVICE = {
    "Bacterial_spot": "Apply copper-based fungicides. Ensure adequate plant spacing to facilitate airflow and avoid overhead watering.",
    "Early_blight": "Remove lower infected leaves. Apply chlorothalonil or copper fungicides. Practice crop rotation.",
    "Late_blight": "Apply systemic fungicides (e.g. mancozeb or copper octanoate) immediately. Destroy infected crop debris.",
    "Leaf_Mold": "Improve greenhouse/field ventilation. Reduce humidity levels and apply appropriate protective fungicides.",
    "Septoria_leaf_spot": "Avoid splashing water on leaves. Mulch soil around plants and apply organic or synthetic fungicides.",
    "Spider_mites": "Spray with insecticidal soap or neem oil. Maintain soil moisture as spider mites thrive in dry conditions.",
    "Target_Spot": "Remove crop residue post-harvest. Apply recommended foliar fungicides early upon disease detection.",
    "Yellow_Leaf_Curl_Virus": "Control whitefly vector population using sticky traps or insect netting. Remove infected plants.",
    "Mosaic_virus": "No direct cure. Rogue and destroy infected plants immediately to stop vector transmission. Disinfect tools.",
    "healthy": "Crop shows strong physiological health! Maintain regular irrigation, balanced fertilization, and routine monitoring.",
    "Apple_scab": "Apply fungicides (captan or myclobutanil) at green tip stage. Rake and destroy fallen leaves in autumn.",
    "Black_rot": "Prune infected branches. Remove mummified fruit. Apply captan or myclobutanil during bloom.",
    "Cedar_apple_rust": "Remove nearby juniper/cedar hosts. Apply fungicides at pink bud and petal fall stages.",
    "Powdery_mildew": "Apply sulfur or potassium bicarbonate sprays. Improve air circulation around plants.",
    "Common_rust": "Plant resistant hybrids. Apply foliar fungicides if infection occurs before tasseling.",
    "Northern_Leaf_Blight": "Use resistant varieties. Apply foliar fungicides (azoxystrobin) at early infection stages.",
    "Cercospora": "Rotate crops. Apply foliar fungicides. Remove infected debris after harvest.",
    "Esca": "Prune infected wood. No curative treatment — manage by protecting pruning wounds.",
    "Leaf_scorch": "Ensure adequate watering. Mulch to retain soil moisture. Avoid salt-based fertilizers near roots.",
    "Haunglongbing": "No cure exists. Remove and destroy infected trees immediately. Control psyllid vector population.",
    "Citrus_greening": "No cure exists. Remove and destroy infected trees immediately. Control psyllid vector population.",
}

class CropDiseasePredictor:
    """
    Inference Engine for single image leaf disease diagnosis.
    
    Supports:
    - Flat and hierarchical model architectures (auto-detected from checkpoint)
    - Temperature-scaled calibrated probabilities
    - OOD (out-of-distribution) confidence detection via entropy
    """
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing Predictor on device: {self.device}")
        
        # Load model using unified loader (handles flat + hierarchical)
        self.model, checkpoint = load_model_from_checkpoint(checkpoint_path, self.device)
        self.model_type = checkpoint.get("model_type", "resnet18")
        self.class_names = checkpoint["class_names"]
        self.temperature = checkpoint.get("temperature", 1.0)
        self.hierarchical = checkpoint.get("hierarchical", False)
        
        if self.temperature != 1.0:
            print(f"  Temperature scaling: T={self.temperature:.4f}")
        if self.hierarchical:
            print(f"  Hierarchical mode: {checkpoint.get('num_plants', '?')} plant species")

        # Inference Transformation Pipeline (deterministic only — no random augmentations)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, image_input: Union[str, Path, Image.Image], top_k: int = 3,
                confidence_threshold: float = 60.0) -> Dict:
        """
        Runs inference on single image with OOD detection.
        
        Returns dict containing:
        - top_prediction: class name
        - top_confidence: confidence percentage
        - top_k_predictions: list of {class, confidence}
        - treatment_advice: farming guidance
        - is_low_confidence: bool (True if OOD/uncertain)
        - entropy: raw prediction entropy
        - normalized_entropy: entropy in [0, 1] (0=certain, 1=uniform)
        """
        # Safety: ensure model is in eval mode every call (prevents stochastic
        # behavior from Dropout and uses BatchNorm running stats)
        self.model.eval()
        
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image object.")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            
            # Temperature scaling for calibrated probabilities
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            probs = F.softmax(logits, dim=1)[0]

        # OOD detection: compute prediction entropy
        # H = -sum(p * log(p)), normalized by max possible entropy log(N)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = torch.log(torch.tensor(len(self.class_names), dtype=torch.float)).item()
        normalized_entropy = entropy / max_entropy  # 0 = certain, 1 = uniform random

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))

        results = []
        for prob, idx in zip(top_probs, top_indices):
            c_name = self.class_names[idx.item()]
            conf = float(prob.item()) * 100.0
            results.append({"class": c_name, "confidence": round(conf, 2)})

        top_class = results[0]["class"]
        top_conf = results[0]["confidence"]
        
        # Flag low confidence / OOD predictions
        is_low_confidence = top_conf < confidence_threshold or normalized_entropy > 0.5
        
        # Fetch agricultural advice matching key substring
        advice = "Consult local agricultural extension officer for specific localized chemical/organic control."
        for key, text in TREATMENT_ADVICE.items():
            if key.lower() in top_class.lower():
                advice = text
                break

        return {
            "top_prediction": top_class,
            "top_confidence": top_conf,
            "top_k_predictions": results,
            "treatment_advice": advice,
            "is_low_confidence": is_low_confidence,
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predictor = CropDiseasePredictor(args.checkpoint)
    result = predictor.predict(args.image)
    print("\nDIAGNOSIS RESULTS:")
    print(f"Predicted Disease: {result['top_prediction']}")
    print(f"Confidence:        {result['top_confidence']}%")
    print(f"Low Confidence:    {result['is_low_confidence']}")
    print(f"Entropy:           {result['normalized_entropy']:.3f}")
    print(f"Treatment Advice:  {result['treatment_advice']}\n")
