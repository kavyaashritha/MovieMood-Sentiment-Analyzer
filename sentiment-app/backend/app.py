import json
import os
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Local import for fallback training if artifacts missing
try:
	# When running as a package: `python -m sentiment-app.backend.app`
	from .model_training import clean_text, TrainConfig, train_and_evaluate  # type: ignore
except Exception:
	# Fallback when running as script inside backend dir: `python app.py`
	from model_training import clean_text, TrainConfig, train_and_evaluate  # type: ignore

BACKEND_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BACKEND_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(BACKEND_DIR, "vectorizer.joblib")
IMAGE_MODEL_PATH = os.path.join(BACKEND_DIR, "image_model.h5")
IMAGE_LABELS_PATH = os.path.join(BACKEND_DIR, "labels.json")

app = FastAPI(title="MovieMood — Sentiment Analyzer", version="1.0.0")

# CORS for local frontend dev (Vite default: 5173)
app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:5173",
		"http://127.0.0.1:5173",
		"*",  # Relax for demo; tighten in production
	],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class PredictRequest(BaseModel):
	review: str


class PredictResponse(BaseModel):
	prediction: str


def _load_or_train_artifacts():
	"""
	Load model/vectorizer or train them if missing for a self-contained demo.
	"""
	if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
		cfg = TrainConfig(
			data_csv_path=os.path.join(BACKEND_DIR, "imdb_dataset.csv"),
			model_out_path=MODEL_PATH,
			vectorizer_out_path=VECTORIZER_PATH,
		)
		train_and_evaluate(cfg)
	model = joblib.load(MODEL_PATH)
	vectorizer = joblib.load(VECTORIZER_PATH)
	return model, vectorizer


MODEL, VECTORIZER = _load_or_train_artifacts()

# Image model loading (lazy load)
IMAGE_MODEL = None
IMAGE_LABELS = None


def _load_image_model():
	"""
	Lazy load image model and labels if available.
	"""
	global IMAGE_MODEL, IMAGE_LABELS
	if IMAGE_MODEL is None:
		if os.path.exists(IMAGE_MODEL_PATH) and os.path.exists(IMAGE_LABELS_PATH):
			try:
				from tensorflow import keras
				IMAGE_MODEL = keras.models.load_model(IMAGE_MODEL_PATH)
				with open(IMAGE_LABELS_PATH, "r") as f:
					IMAGE_LABELS = json.load(f)
				print("Image model loaded successfully")
			except Exception as e:
				print(f"Warning: Could not load image model: {e}")
				IMAGE_MODEL = None
				IMAGE_LABELS = None
	return IMAGE_MODEL, IMAGE_LABELS


def preprocess_image(image_file: UploadFile) -> np.ndarray:
	"""
	Preprocess uploaded image: resize to 224x224, normalize to [0,1].
	"""
	img = Image.open(image_file.file)
	if img.mode != "RGB":
		img = img.convert("RGB")
	img = img.resize((224, 224))
	img_array = np.array(img, dtype=np.float32) / 255.0
	return np.expand_dims(img_array, axis=0)  # Add batch dimension


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
	text = payload.review or ""
	if not text.strip():
		raise HTTPException(status_code=400, detail="review must be a non-empty string")
	cleaned = clean_text(text)
	vec = VECTORIZER.transform([cleaned])
	pred = MODEL.predict(vec)[0]
	label = "positive" if int(pred) == 1 else "negative"
	return PredictResponse(prediction=label)


@app.post("/predict-image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
	"""
	Predict emotion/sentiment from uploaded image.
	Accepts: JPG, PNG, JPEG, BMP
	Returns: { "prediction": "<emotion_label>" }
	"""
	# Check file type
	if not file.content_type or not file.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="File must be an image")
	
	# Load image model
	model, labels = _load_image_model()
	if model is None or labels is None:
		raise HTTPException(
			status_code=503,
			detail="Image model not available. Please train the model first using train_image_model.py"
		)
	
	try:
		# Preprocess image
		img_array = preprocess_image(file)
		
		# Predict
		predictions = model.predict(img_array, verbose=0)
		predicted_idx = int(np.argmax(predictions[0]))
		
		# Map index to label
		label = labels.get(str(predicted_idx), "unknown")
		
		return PredictResponse(prediction=label)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)


