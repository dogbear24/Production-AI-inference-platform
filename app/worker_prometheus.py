from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)


WORKER_JOBS_PROCESSED = Counter(
    "aip_worker_jobs_processed",
    "Total inference jobs processed by the worker",
    ["status"],
)


WORKER_JOB_DURATION = Histogram(
    "aip_worker_job_duration_seconds",
    "Total worker processing time per job",
)


WORKER_INFERENCE_DURATION = Histogram(
    "aip_worker_inference_duration_seconds",
    "Pure model inference duration",
)


WORKER_QUEUE_DEPTH = Gauge(
    "aip_worker_queue_depth",
    "Current number of jobs waiting in Redis",
)

WORKER_BATCH_SIZE = Histogram(
    "aip_worker_batch_size",
    "Number of jobs processed in each inference batch",
    buckets=(
        1,
        2,
        4,
        8,
        16,
        32,
    ),
)


WORKER_BATCH_DURATION = Histogram(
    "aip_worker_batch_duration_seconds",
    "Total time required to process an inference batch",
)