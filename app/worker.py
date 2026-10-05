import json
import time

import redis

from app.inference import ImageClassifier
from app.redis_queue import (
    JOB_TTL_SECONDS,
    QUEUE_NAME,
    redis_binary,
    redis_text,
    utc_now,
)


classifier = ImageClassifier()


def process_job(job_id):

    job_key = f"job:{job_id}"

    input_key = (
        f"job:{job_id}:input"
    )

    print(
        f"[worker] Processing {job_id}"
    )

    try:

        metadata = redis_text.hgetall(
            job_key
        )

        if not metadata:

            print(
                f"[worker] Job {job_id} "
                "does not exist."
            )

            return

        redis_text.hset(
            job_key,
            mapping={
                "status": "processing",
                "started_at": utc_now(),
            },
        )

        image_bytes = None

        for attempt in range(3):

            image_bytes = redis_binary.get(
                input_key
            )

            if image_bytes is not None:
                break

            time.sleep(0.25)


        if image_bytes is None:

            raise RuntimeError(
                "Input image expired "
                "or could not be found."
            )

        top_k = int(
            metadata.get(
                "top_k",
                "5",
            )
        )

        result = classifier.predict(
            image_bytes,
            top_k=top_k,
        )

        redis_text.hset(
            job_key,
            mapping={
                "status": "completed",
                "finished_at": utc_now(),
                "device":
                    classifier.device_name,
                "result":
                    json.dumps(result),
            },
        )

        redis_text.expire(
            job_key,
            JOB_TTL_SECONDS,
        )

        redis_binary.delete(
            input_key
        )

        print(
            f"[worker] Completed {job_id}"
        )

    except Exception as exc:

        redis_text.hset(
            job_key,
            mapping={
                "status": "failed",
                "finished_at": utc_now(),
                "error": str(exc),
            },
        )

        redis_text.expire(
            job_key,
            JOB_TTL_SECONDS,
        )

        print(
            f"[worker] Failed {job_id}: "
            f"{exc}"
        )


def run_worker():

    print(
        "=============================="
    )

    print(
        " AI Inference Worker"
    )

    print(
        "=============================="
    )

    print(
        f"Device: {classifier.device_name}"
    )

    print(
        f"Queue: {QUEUE_NAME}"
    )

    print(
        "Waiting for jobs..."
    )

    while True:

        try:

            item = redis_text.blpop(
                QUEUE_NAME,
                timeout=5,
            )

            if item is None:
                continue

            _, job_id = item

            process_job(
                job_id
            )

        except redis.ConnectionError:

            print(
                "[worker] Redis unavailable. "
                "Retrying in 3 seconds..."
            )

            time.sleep(3)

        except KeyboardInterrupt:

            print(
                "\n[worker] Shutting down."
            )

            break


if __name__ == "__main__":
    run_worker()