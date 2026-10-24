"""Local face/model analysis.

Optional trained-model contract:
  Put a fine-tuned binary classifier at models/deepfake_efficientnet_b0.pth.
  It must be a torchvision EfficientNet-B0 state_dict with its final classifier
  changed to 2 outputs ordered [real, fake]. This file deliberately does NOT
  treat ImageNet weights as a deepfake detector.
"""
from pathlib import Path
from io import BytesIO
import base64
import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = Path(__file__).parent / "models" / "deepfake_efficientnet_b0.pth"

def _png64(image):
    buf = BytesIO(); image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def analyze_image(image):
    face_count = 0
    # Safe fallback visual: draw detected face boxes if MTCNN is installed.
    marked = image.convert("RGB").copy()
    try:
        from facenet_pytorch import MTCNN
        import torch
        mtcnn = MTCNN(keep_all=True, device="cuda:0" if torch.cuda.is_available() else "cpu")
        boxes, _ = mtcnn.detect(image.convert("RGB"))
        if boxes is not None:
            face_count = len(boxes)
            draw = ImageDraw.Draw(marked)
            for box in boxes:
                draw.rectangle(tuple(map(float, box)), outline=(255, 255, 0), width=3)
    except Exception:
        # Face detection is optional; keep the API functional without model downloads.
        pass

    if MODEL_PATH.exists():
        try:
            import torch
            from torchvision.models import efficientnet_b0
            from torchvision import transforms
            model = efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            fake_score = float(probs[1])
            return {
                "verdict": "likely_fake" if fake_score >= 0.5 else "likely_real",
                "confidence": float(max(fake_score, 1 - fake_score)),
                "ai_detection_score": fake_score,
                "model_status": "trained_deepfake_model",
                "face_count": face_count,
                "gradcam_png_base64": _png64(marked)
            }
        except Exception as exc:
            model_status = "model_load_error: " + str(exc)[:180]
    else:
        model_status = "no_trained_deepfake_model"

    return {
        "verdict": "inconclusive",
        "confidence": None,
        "ai_detection_score": None,
        "model_status": model_status,
        "face_count": face_count,
        "gradcam_png_base64": _png64(marked)
    }
