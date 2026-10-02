from time import perf_counter
from uuid import uuid4
from app.database import init_db
from app.registry_routes import router as registry_router

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile
)

from app.inference import ImageClassifier
from app.metrics import MetricsCollector


app = FastAPI(

    title="AI Inference Platform",

    version="0.4.0",

    description=(
        "Production-oriented "
        "machine learning inference API"
    )
)

init_db()

app.include_router(registry_router)


classifier = ImageClassifier()

metrics = MetricsCollector()


# ==================================
# REQUEST METRICS MIDDLEWARE
# ==================================

@app.middleware("http")
async def request_metrics_middleware(
    request: Request,
    call_next
):

    request_id = uuid4().hex

    request.state.request_id = request_id

    start = perf_counter()

    try:

        response = await call_next(request)

        status_code = response.status_code

    except Exception:

        latency_ms = (
            perf_counter() - start
        ) * 1000

        metrics.record_request(
            latency_ms,
            500
        )

        raise

    latency_ms = (
        perf_counter() - start
    ) * 1000

    metrics.record_request(
        latency_ms,
        status_code
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Response-Time-Ms"
    ] = f"{latency_ms:.3f}"

    return response


# ==================================
# ROOT
# ==================================

@app.get("/")
def root():

    return {

        "name":
            "AI Inference Platform",

        "version":
            "0.4.0",

        "docs":
            "/docs"
    }


# ==================================
# HEALTH
# ==================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "model_loaded":
            classifier.is_loaded,

        "device":
            classifier.device_name
    }


# ==================================
# METRICS
# ==================================

@app.get("/metrics")
def get_metrics():

    return {

        "api":
            metrics.snapshot(),

        "model": {

            "loaded":
                classifier.is_loaded,

            "device":
                classifier.device_name,

            "last_model_load_ms":
                round(
                    classifier.last_model_load_ms,
                    3
                )
        }
    }


# ==================================
# PREDICT
# ==================================

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 5
):

    if top_k < 1 or top_k > 10:

        raise HTTPException(
            status_code=400,
            detail=(
                "top_k must be "
                "between 1 and 10"
            )
        )

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:

        raise HTTPException(
            status_code=415,
            detail=(
                "Upload a JPEG, "
                "PNG, or WebP image."
            )
        )

    contents = await file.read()

    try:

        result = classifier.predict(
            contents,
            top_k=top_k
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Inference failed."
        ) from exc

    return {

        "request_id":
            request.state.request_id,

        "filename":
            file.filename,

        "device":
            classifier.device_name,

        **result
    }