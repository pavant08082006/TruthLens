# TruthLens Local MVP

A local-first prototype for image/video forensic triage. No cloud API is used.

## Important model limitation
The project runs end-to-end without model weights, but it will return **INCONCLUSIVE** for the fake/real verdict until you provide a genuinely fine-tuned deepfake classifier. ImageNet-pretrained EfficientNet weights are not a deepfake detector, so this app does not misrepresent them as one. ELA is a compression-difference visualization, not proof of tampering.

The Grad-CAM dependency is included for the next model-integration step. The current visual panel displays detected face boxes (when MTCNN works); it is not falsely labelled as a Grad-CAM heatmap unless a proper CAM implementation is connected to a trained model.

## Requirements
- Python 3.10 or 3.11 recommended
- FFmpeg is not required; OpenCV decodes MP4 files.
- Internet may be needed once to install Python packages. Runtime analysis is local.

## Setup (Windows PowerShell)
```powershell
cd truthlens_local_mvp
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run backend
Open terminal 1:
```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```
Check `http://127.0.0.1:8000/health` or API docs at `http://127.0.0.1:8000/docs`.

## Run dashboard
Open terminal 2 in the same folder:
```powershell
streamlit run app.py
```
Open the local address Streamlit prints, usually `http://localhost:8501`.

## API test
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/analyze" -F "file=@sample.jpg"
```
For MP4:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/analyze" -F "file=@sample.mp4"
```

## Add a trained classifier
Place `models/deepfake_efficientnet_b0.pth` in this project. It must be a PyTorch `state_dict` for torchvision EfficientNet-B0 with the final classifier replaced by a 2-output layer; output index 0 must mean `real`, index 1 must mean `fake`. It should be trained/validated on a suitable deepfake dataset, with documented test-set metrics. Do not use a random checkpoint or ImageNet-only model as a fake detector.

## Current deliverables
- `main.py`: FastAPI `/api/v1/analyze`, image validation, 1 FPS video sampling, structured JSON.
- `ela.py`: ELA map and numeric summary.
- `detector.py`: optional MTCNN face boxes and loading contract for a trained classifier.
- `app.py`: Streamlit upload, evidence panels, and ZIP report export.

## Known limitations
- MTCNN may download pretrained weights on first use depending on package/cache configuration. For fully offline operation, pre-cache its weights or run without `facenet-pytorch`; the API still functions.
- Current dashboard uses face-box visualization rather than a true Grad-CAM heatmap. Grad-CAM needs to be wired to a trained model and target layer; this is intentionally not faked.
- ELA is meaningful primarily for JPEG-like compression analysis. PNG uploads are converted to JPEG for the comparison.
- Video decoding depends on your local OpenCV build and the uploaded MP4 codec.
