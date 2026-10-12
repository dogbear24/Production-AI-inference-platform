import json
import os
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
from prometheus_client import start_http_server

from app.worker_prometheus import (
    WORKER_BATCH_DURATION,
    WORKER_BATCH_SIZE,
    WORKER_INFERENCE_DURATION,
    WORKER_JOB_DURATION,
    WORKER_JOBS_PROCESSED,
    WORKER_QUEUE_DEPTH,
)

from app.onnx_inference import (
    ONNXImageClassifier,
)


INFERENCE_BACKEND = os.getenv(
    "INFERENCE_BACKEND",
    "pytorch",
).lower()


if INFERENCE_BACKEND == "onnx":

    classifier = (
        ONNXImageClassifier()
    )

elif INFERENCE_BACKEND == "pytorch":

    classifier = (
        ImageClassifier()
    )

else:

    raise RuntimeError(
        "Unknown inference backend: "
        f"{INFERENCE_BACKEND}"
    )


MAX_BATCH_SIZE = int(
    os.getenv(
        "MAX_BATCH_SIZE",
        "8",
    )
)


BATCH_WAIT_MS = float(
    os.getenv(
        "BATCH_WAIT_MS",
        "25",
    )
)


def process_job(job_id):

    job_started = time.perf_counter()

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

        WORKER_INFERENCE_DURATION.observe(
            result["inference_ms"] / 1000
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

        WORKER_JOBS_PROCESSED.labels(
            status="failed"
        ).inc()

        WORKER_JOB_DURATION.observe(
            time.perf_counter()
            - job_started
        )


def collect_batch(
    first_job_id: str,
) -> list[str]:

    job_ids = [
        first_job_id
    ]

    deadline = (
        time.perf_counter()
        + BATCH_WAIT_MS / 1000
    )

    while (
        len(job_ids)
        < MAX_BATCH_SIZE
    ):

        remaining_slots = (
            MAX_BATCH_SIZE
            - len(job_ids)
        )

        extra_job_ids = redis_text.lpop(
            QUEUE_NAME,
            remaining_slots,
        )

        if extra_job_ids:

            if isinstance(
                extra_job_ids,
                str,
            ):
                extra_job_ids = [
                    extra_job_ids
                ]

            job_ids.extend(
                extra_job_ids
            )

            continue

        if (
            time.perf_counter()
            >= deadline
        ):
            break

        time.sleep(0.001)

    return job_ids


def process_batch(
    job_ids: list[str],
):

    batch_started = (
        time.perf_counter()
    )

    valid_jobs = []

    # ==================================
    # LOAD JOB DATA
    # ==================================

    for job_id in job_ids:

        job_key = (
            f"job:{job_id}"
        )

        input_key = (
            f"job:{job_id}:input"
        )

        try:

            metadata = (
                redis_text.hgetall(
                    job_key
                )
            )

            if not metadata:

                print(
                    "[worker] Missing job "
                    f"{job_id}"
                )

                continue

            redis_text.hset(
                job_key,
                mapping={
                    "status":
                        "processing",

                    "started_at":
                        utc_now(),
                },
            )

            image_bytes = None

            for _ in range(3):

                image_bytes = (
                    redis_binary.get(
                        input_key
                    )
                )

                if (
                    image_bytes
                    is not None
                ):
                    break

                time.sleep(0.25)

            if image_bytes is None:

                raise RuntimeError(
                    "Input image expired "
                    "or could not be found."
                )

            valid_jobs.append({
                "job_id":
                    job_id,

                "job_key":
                    job_key,

                "input_key":
                    input_key,

                "image_bytes":
                    image_bytes,

                "top_k":
                    int(
                        metadata.get(
                            "top_k",
                            "5",
                        )
                    ),
            })

        except Exception as exc:

            redis_text.hset(
                job_key,
                mapping={
                    "status":
                        "failed",

                    "finished_at":
                        utc_now(),

                    "error":
                        str(exc),
                },
            )

            WORKER_JOBS_PROCESSED.labels(
                status="failed"
            ).inc()

            print(
                f"[worker] Failed "
                f"{job_id}: {exc}"
            )

    if not valid_jobs:
        return

    # ==================================
    # ONE BATCH INFERENCE
    # ==================================

    try:

        image_bytes_list = [
            job["image_bytes"]
            for job in valid_jobs
        ]

        top_ks = [
            job["top_k"]
            for job in valid_jobs
        ]

        batch_result = (
            classifier.predict_batch(
                image_bytes_list,
                top_ks,
            )
        )

        WORKER_BATCH_SIZE.observe(
            len(valid_jobs)
        )

        WORKER_INFERENCE_DURATION.observe(
            batch_result[
                "inference_ms"
            ]
            / 1000
        )

        # ==================================
        # SPLIT RESULTS BACK INTO JOBS
        # ==================================

        for index, job in enumerate(
            valid_jobs
        ):

            individual_result = {
                "batch_size":
                    batch_result[
                        "batch_size"
                    ],

                "model_load_ms":
                    batch_result[
                        "model_load_ms"
                    ],

                "preprocess_ms":
                    batch_result[
                        "preprocess_ms"
                    ],

                "inference_ms":
                    batch_result[
                        "inference_ms"
                    ],

                "predictions":
                    batch_result[
                        "predictions"
                    ][index],
            }

            redis_text.hset(
                job["job_key"],
                mapping={
                    "status":
                        "completed",

                    "finished_at":
                        utc_now(),

                    "device":
                        classifier.device_name,

                    "result":
                        json.dumps(
                            individual_result
                        ),
                },
            )

            redis_text.expire(
                job["job_key"],
                JOB_TTL_SECONDS,
            )

            redis_binary.delete(
                job["input_key"]
            )

            WORKER_JOBS_PROCESSED.labels(
                status="completed"
            ).inc()

            print(
                "[worker] Completed "
                f"{job['job_id']} "
                f"(batch="
                f"{len(valid_jobs)})"
            )

    except Exception as exc:

        for job in valid_jobs:

            redis_text.hset(
                job["job_key"],
                mapping={
                    "status":
                        "failed",

                    "finished_at":
                        utc_now(),

                    "error":
                        str(exc),
                },
            )

            WORKER_JOBS_PROCESSED.labels(
                status="failed"
            ).inc()

        print(
            "[worker] Batch failed: "
            f"{exc}"
        )

    finally:

        WORKER_BATCH_DURATION.observe(
            time.perf_counter()
            - batch_started
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

    classifier.load()

    print(
        f"Device: {classifier.device_name}"
    )

    print(
        f"Queue: {QUEUE_NAME}"
    )

    print(
        "Waiting for jobs..."
    )

    start_http_server(8001)

    print(
        "Prometheus metrics: "
        "http://0.0.0.0:8001/metrics"
    )

    print(
        "Inference backend:",
        INFERENCE_BACKEND,
    )

    classifier.load()

    print(
        "Device / provider:",
        classifier.device_name,
    )

    while True:

        try:

            WORKER_QUEUE_DEPTH.set(
                redis_text.llen(
                    QUEUE_NAME
                )
            )

        except redis.RedisError:

            WORKER_QUEUE_DEPTH.set(0)

        try:

            item = redis_text.blpop(
                QUEUE_NAME,
                timeout=5,
            )

            if item is None:
                continue

            _, first_job_id = item

            job_ids = collect_batch(
                first_job_id
            )

            WORKER_QUEUE_DEPTH.set(
                redis_text.llen(
                    QUEUE_NAME
                )
            )

            print(
                "[worker] Processing batch "
                f"of {len(job_ids)} job(s)"
            )

            process_batch(
                job_ids
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
