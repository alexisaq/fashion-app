from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fashion-MNIST API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Cargando modelo...")
pipeline = joblib.load("backend/pipeline.pkl")
print("Modelo listo.")

class ImageData(BaseModel):
    pixels: list[float]  # lista de 784 valores 0-255

@app.get("/")
def root():
    return {"status": "ok", "message": "Fashion-MNIST API activa"}

@app.post("/predict")
def predict(data: ImageData):
    if len(data.pixels) != 784:
        raise HTTPException(status_code=400, detail="Se necesitan exactamente 784 píxeles")

    X = np.array(data.pixels).reshape(1, -1)
    pred_class = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0]

    top3 = sorted(
        [{"clase": CLASS_NAMES[i], "confianza": round(float(p) * 100, 1)} for i, p in enumerate(proba)],
        key=lambda x: x["confianza"], reverse=True
    )[:3]

    return {
        "prediccion": CLASS_NAMES[pred_class],
        "confianza": round(float(proba[pred_class]) * 100, 1),
        "top3": top3
    }