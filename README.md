# DermaScan - A Skin-Analysis Web Application

## Folder structure
```
acne-app/
├── main.py                  # FastAPI backend
├── requirements.txt
├── acne_model_best.h5       # ← copy from Google Drive
├── class_names.json         # ← export from Colab (see below)
└── templates/
    └── index.html           # Frontend
```

## Step 1 — Export class_names.json from Colab
Run this in your notebook after Step Pre-16:
```python
import json
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
```
Download it (Files panel → right-click → Download),
then place it in the acne-app folder.

## Step 2 — Download your model from Drive
Download `acne_model_best.h5` from:
`MyDrive/Acne_Dataset/acne_model_best.h5`
Place it in the acne-app folder.

## Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

## Step 4 — Run
```bash
uvicorn main:app --reload
```
Open http://localhost:8000 in your browser.

## On mobile (for the demo)
Run the server on your laptop, then on your phone open:
`http://<your-laptop-ip>:8000`
(Find your IP with `ipconfig` on Windows or `ifconfig` on Mac/Linux)
The upload button will trigger the phone camera directly.

## Giving to Cursor
Open the acne-app folder in Cursor.
Suggested prompt to improve the UI or add features:
> "This is a FastAPI + HTML acne diagnosis app. The backend is in main.py
>  and the frontend in templates/index.html. The /predict endpoint accepts
>  an image upload and returns predicted_class, confidence, severity,
>  description, recommendation, and all_confidences as JSON.
>  [describe what you want changed]"
