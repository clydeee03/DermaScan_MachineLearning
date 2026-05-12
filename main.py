"""
Acne Diagnosis Web App — FastAPI Backend
========================================
Setup:
    pip install fastapi uvicorn python-multipart jinja2 tensorflow pillow numpy

Run:
    uvicorn main:app --reload

Then open http://localhost:8000 in your browser.

Place these files in the same folder as main.py:
    acne_model_best.h5
    class_names.json
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH      = Path("acne_model_best.h5")
CLASS_NAMES_PATH = Path("class_names.json")

SEVERITY_MAPPING = {
    'Clear'    : 'None',
    'Blackhead': 'Mild',
    'Whitehead': 'Mild',
    'Papule'   : 'Moderate',
    'Pustule'  : 'Moderate',
    'Nodule'   : 'Severe',
    'Cyst'     : 'Severe',
    'Scar'     : 'Post-Acne',
}

RECOMMENDATIONS = {
    'None'     : "Your skin looks clear! Keep up your current skincare routine.",
    'Mild'     : "Mild acne detected. Try a gentle cleanser with salicylic acid. Avoid touching your face.",
    'Moderate' : "Moderate acne detected. Consider seeing a dermatologist for a tailored treatment plan.",
    'Severe'   : "Severe acne detected. Please consult a dermatologist as soon as possible.",
    'Post-Acne': "Post-acne scarring detected. Ingredients like niacinamide and retinol may help. A dermatologist can advise on stronger treatments.",
}

ACNE_DESCRIPTIONS = {
    'Clear'    : "No active acne detected. Your skin appears healthy.",
    'Blackhead': "Open comedone — a pore clogged with oxidized oil that appears dark on the surface.",
    'Whitehead': "Closed comedone — a pore clogged beneath the skin surface, appearing as a small flesh-colored bump.",
    'Papule'   : "A small, inflamed red bump with no visible pus. Tender to the touch.",
    'Pustule'  : "An inflamed bump with a visible white or yellow pus tip — commonly called a pimple.",
    'Nodule'   : "A large, deep, hard lump beneath the skin surface. Often painful.",
    'Cyst'     : "A large, deep, pus-filled lump — the most severe form of active acne. Can cause scarring.",
    'Scar'     : "Post-acne marks left over from previous acne lesions. Can be atrophic (indented) or hypertrophic (raised).",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load model + class names once at startup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermaScan: A Skin Analysis Tool",
    description="Upload a skin photo for AI-assisted acne class prediction (informational only).",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

model       = None
class_names = None

@app.on_event("startup")
def load_model():
    global model, class_names

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. "
                           "Place acne_model_best.h5 in the same folder as main.py.")
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(f"Class names file not found: {CLASS_NAMES_PATH}. "
                           "Place class_names.json in the same folder as main.py.")

    print("Loading model...")
    model = keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    print(f"Model loaded. Classes: {class_names}")


static_dir = Path("static")
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)

    preds      = model.predict(arr)[0]
    idx        = int(np.argmax(preds))
    pred_class = class_names[idx]
    confidence = float(preds[idx]) * 100
    severity   = SEVERITY_MAPPING.get(pred_class, "Unknown")

    # All class confidences for the chart
    all_confidences = {class_names[i]: round(float(preds[i]) * 100, 2)
                       for i in range(len(class_names))}

    return {
        "predicted_class" : pred_class,
        "confidence"      : round(confidence, 1),
        "severity"        : severity,
        "description"     : ACNE_DESCRIPTIONS.get(pred_class, ""),
        "recommendation"  : RECOMMENDATIONS.get(severity, "Please consult a dermatologist."),
        "all_confidences" : all_confidences,
    }


def _is_image_bytes(data: bytes) -> bool:
    """Validate image bytes (works when mobile sends application/octet-stream)."""
    if not data:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the single-page web UI."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"title": "DermaScan - A Skin Analysis Tool"},
    )


@app.get("/health")
def health():
    """Liveness check for hosting platforms."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs the model, returns a JSON diagnosis.
    """
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not _is_image_bytes(image_bytes):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid image (e.g. JPG or PNG).",
        )

    try:
        result = run_prediction(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return result
