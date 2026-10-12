from prometheus_client import Counter, Histogram


HTTP_REQUESTS = Counter(
    "aip_http_requests",
    "Total HTTP requests handled by the API",
    ["method", "status"],
)


HTTP_REQUEST_LATENCY = Histogram(
    "aip_http_request_duration_seconds",
    "End-to-end HTTP request latency",
    ["method"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)


INFERENCE_JOBS_SUBMITTED = Counter(
    "aip_inference_jobs_submitted",
    "Total asynchronous inference jobs submitted",
)