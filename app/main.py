
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.inference import ImageClassifier

app = FastAPI(
    title="AI Inference Platform",
    version="0.1.0",
    description="A starter ML model-serving API built with FastAPI and PyTorch.",
)

classifier = ImageClassifier()

@app.get("/")
def root():
    return {
        "name": "AI Inference Platform",
        "version": "0.1.0",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier.is_loaded,
        "device": classifier.device_name,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    contents = await file.read()

    try:
        predictions = classifier.predict(contents, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    return {
        "filename": file.filename,
        "device": classifier.device_name,
        "predictions": predictions,
    }
