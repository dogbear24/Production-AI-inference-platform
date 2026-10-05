import redis

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.redis_queue import (
    enqueue_job,
    get_job,
    queue_size,
    redis_available,
)


router = APIRouter(
    tags=["inference jobs"],
)


@router.get(
    "/queue/health"
)
def queue_health():

    connected = (
        redis_available()
    )

    return {
        "redis_connected":
            connected,

        "queue_size":
            queue_size()
            if connected
            else None,
    }


@router.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(
    file: UploadFile = File(...),
    top_k: int = 5,
):

    if top_k < 1 or top_k > 10:

        raise HTTPException(
            status_code=400,
            detail=(
                "top_k must be "
                "between 1 and 10"
            ),
        )

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:

        raise HTTPException(
            status_code=415,
            detail=(
                "Upload a JPEG, PNG, "
                "or WebP image."
            ),
        )

    if not redis_available():

        raise HTTPException(
            status_code=503,
            detail=(
                "Redis queue "
                "is unavailable."
            ),
        )

    contents = await file.read()

    try:

        job_id = enqueue_job(
            filename=file.filename,
            content_type=file.content_type,
            image_bytes=contents,
            top_k=top_k,
        )

    except redis.RedisError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not enqueue job."
            ),
        ) from exc

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url":
            f"/jobs/{job_id}",
    }


@router.get(
    "/jobs/{job_id}"
)
def job_status(
    job_id: str
):

    try:

        job = get_job(
            job_id
        )

    except redis.RedisError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Redis queue "
                "is unavailable."
            ),
        ) from exc

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return job