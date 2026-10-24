from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
import numpy as np, base64

def create_ela(image, quality=90):
    """Create an ELA map and basic descriptive metrics."""
    original = image.convert("RGB")
    buffer = BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(original, recompressed)
    arr = np.asarray(diff, dtype=np.float32)
    peak = float(arr.max()) if arr.size else 0.0
    scale = 255.0 / max(peak, 1.0)
    ela = ImageEnhance.Brightness(diff).enhance(scale)
    diff_gray = np.asarray(diff.convert("L"), dtype=np.float32)
    mean_diff = float(diff_gray.mean())
    high_fraction = float(np.mean(diff_gray >= 20))
    out = BytesIO()
    ela.save(out, format="PNG")
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    notes = "Compression differences are present; inspect visually." if high_fraction > 0.05 else "Low-to-moderate ELA differences; this does not establish authenticity."
    return ela, {"mean_difference": mean_diff, "high_difference_fraction": high_fraction, "notes": notes, "png_base64": encoded}
