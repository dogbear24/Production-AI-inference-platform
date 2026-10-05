import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import redis


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

QUEUE_NAME = "inference:queue"

JOB_TTL_SECONDS = 3600


redis_text = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=10,
)

redis_binary = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=False,
    socket_connect_timeout=2,
    socket_timeout=10,
)


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def redis_available():

    try:
        return redis_text.ping()

    except redis.RedisError:
        return False


def queue_size():

    try:
        return redis_text.llen(
            QUEUE_NAME
        )

    except redis.RedisError:
        return 0


def enqueue_job(
    filename,
    content_type,
    image_bytes,
    top_k,
):

    job_id = uuid4().hex

    job_key = f"job:{job_id}"

    input_key = f"job:{job_id}:input"

    # IMPORTANT:
    # Store the image BEFORE adding the job
    # to the worker queue.
    redis_binary.set(
        input_key,
        image_bytes,
        ex=JOB_TTL_SECONDS,
    )

    # Create the job metadata.
    redis_text.hset(
        job_key,
        mapping={
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "content_type": content_type,
            "top_k": str(top_k),
            "created_at": utc_now(),
        },
    )

    redis_text.expire(
        job_key,
        JOB_TTL_SECONDS,
    )

    # ALWAYS enqueue last.
    # Once this happens, the worker
    # is allowed to pick up the job.
    redis_text.rpush(
        QUEUE_NAME,
        job_id,
    )

    return job_id


def get_job(job_id):

    job_key = f"job:{job_id}"

    data = redis_text.hgetall(
        job_key
    )

    if not data:
        return None

    result_json = data.get(
        "result"
    )

    result = None

    if result_json:
        result = json.loads(
            result_json
        )

    response = {
        "job_id": job_id,
        "status": data.get(
            "status"
        ),
        "filename": data.get(
            "filename"
        ),
        "created_at": data.get(
            "created_at"
        ),
        "started_at": data.get(
            "started_at"
        ),
        "finished_at": data.get(
            "finished_at"
        ),
        "device": data.get(
            "device"
        ),
        "error": data.get(
            "error"
        ),
        "result": result,
    }

    return response