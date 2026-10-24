from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, shutil, cv2, numpy as np
from PIL import Image
from io import BytesIO
from ela import create_ela
from detector import analyze_image

app = FastAPI(title="TruthLens Local MVP", version="0.1.0")
ALLOWED = {".jpg", ".png", ".mp4"}
MAX_BYTES = 100 * 1024 * 1024

def read_image_bytes(data: bytes):
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

def sample_video(data: bytes, suffix=".mp4"):
    """Decode a video and return at most one frame per second."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = tmp.name
    tmp.write(data); tmp.close()
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else 0
    seconds = max(1, int(np.ceil(duration)))
    for sec in range(seconds):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((sec, Image.fromarray(frame)))
    cap.release()
    Path(path).unlink(missing_ok=True)
    if not frames:
        raise HTTPException(status_code=400, detail="Could not decode video. Ensure OpenCV supports this MP4.")
    return frames

@app.get("/health")
def health():
    return {"status": "ok", "service": "TruthLens Local MVP"}

@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=415, detail="Supported formats: .jpg, .png, .mp4")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 100 MB limit.")

    if suffix == ".mp4":
        frames = sample_video(data, suffix)
    else:
        frames = [(0, read_image_bytes(data))]

    results = []
    for second, image in frames:
        ela_img, ela_metrics = create_ela(image)
        prediction = analyze_image(image)
        results.append({
            "second": second,
            "verdict": prediction["verdict"],
            "overall_confidence": prediction["confidence"],
            "ai_detection_score": prediction["ai_detection_score"],
            "model_status": prediction["model_status"],
            "face_count": prediction["face_count"],
            "forensic_analysis": {
                "ela_mean_difference": ela_metrics["mean_difference"],
                "ela_high_difference_fraction": ela_metrics["high_difference_fraction"],
                "notes": ela_metrics["notes"]
            },
            "visuals": {
                "ela_png_base64": ela_metrics["png_base64"],
                "gradcam_png_base64": prediction["gradcam_png_base64"]
            }
        })
    # Conservative aggregate: never invent a fake probability when no trained detector is installed.
    available = [r for r in results if r["model_status"] == "trained_deepfake_model"]
    if available:
        score = float(np.mean([r["ai_detection_score"] for r in available]))
        verdict = "likely_fake" if score >= 0.5 else "likely_real"
        confidence = max(score, 1 - score)
    else:
        score, confidence, verdict = None, None, "inconclusive"
    return JSONResponse({
        "filename": file.filename,
        "media_type": "video" if suffix == ".mp4" else "image",
        "sampled_frames": len(results),
        "verdict": verdict,
        "overall_confidence": confidence,
        "ai_detection_score": score,
        "forensic_analysis": {
            "frames_analyzed": len(results),
            "ela_mean_difference_average": float(np.mean([r["forensic_analysis"]["ela_mean_difference"] for r in results])),
            "limitations": "ELA indicates compression differences, not proof of manipulation. A trained deepfake model is required for a reliable fake/real score."
        },
        "frame_results": results
    })
