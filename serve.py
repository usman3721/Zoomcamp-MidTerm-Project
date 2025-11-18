from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from predict import load_components, predict

app = FastAPI(title="Language Detection API")

# Load model at startup
model, vectorizer, label_encoder = load_components()

# Pydantic model for request validation
class Texts(BaseModel):
    texts: List[str]

@app.post("/predict")
def predict_languages(data: Texts):
    results = predict(data.texts, model, vectorizer, label_encoder)
    return {"predictions": results.tolist()}

# -----------------------------
# Run with:
# uvicorn serve:app --reload
# -----------------------------
