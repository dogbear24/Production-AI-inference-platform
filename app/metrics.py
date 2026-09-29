from collections import deque
from threading import Lock
from time import perf_counter


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)

    if lower == upper:
        return ordered[lower]

    weight = index - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


class MetricsCollector:

    def __init__(self, max_samples: int = 1000):
        self.latencies = deque(maxlen=max_samples)

        self.total_requests = 0
        self.total_errors = 0

        self.started_at = perf_counter()

        self.lock = Lock()

    def record_request(
        self,
        latency_ms: float,
        status_code: int
    ):
        with self.lock:

            self.total_requests += 1

            if status_code >= 500:
                self.total_errors += 1

            self.latencies.append(latency_ms)

    def snapshot(self):

        with self.lock:

            latencies = list(self.latencies)

            total_requests = self.total_requests
            total_errors = self.total_errors

        uptime_seconds = max(
            perf_counter() - self.started_at,
            0.000001
        )

        average_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else 0
        )

        return {

            "total_requests": total_requests,

            "total_errors": total_errors,

            "samples_stored": len(latencies),

            "average_latency_ms":
                round(average_latency, 3),

            "p50_latency_ms":
                round(percentile(latencies, 0.50), 3),

            "p95_latency_ms":
                round(percentile(latencies, 0.95), 3),

            "p99_latency_ms":
                round(percentile(latencies, 0.99), 3),

            "average_requests_per_second":
                round(
                    total_requests / uptime_seconds,
                    3
                )
        }