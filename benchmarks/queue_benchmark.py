import argparse
import asyncio
import mimetypes
import statistics
from pathlib import Path
from time import perf_counter

import httpx


def percentile(
    values,
    p,
):

    if not values:
        return 0.0

    ordered = sorted(values)

    index = (
        len(ordered) - 1
    ) * p

    lower = int(index)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    if lower == upper:
        return ordered[lower]

    weight = (
        index - lower
    )

    return (
        ordered[lower]
        * (1 - weight)
        + ordered[upper]
        * weight
    )


async def submit_job(
    client,
    semaphore,
    url,
    image_bytes,
    filename,
    content_type,
):
    started = perf_counter()

    async with semaphore:

        response = await client.post(
            f"{url}/jobs",
            files={
                "file": (
                    filename,
                    image_bytes,
                    content_type,
                )
            },
            timeout=60,
        )

        response.raise_for_status()

        return {
            "job_id": response.json()["job_id"],
            "started": started,
        }


async def wait_for_job(
    client,
    url,
    job_id,
    started,
):

    while True:

        response = await client.get(
            f"{url}/jobs/{job_id}",
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        status = data["status"]

        if status == "completed":

            latency_ms = (
                perf_counter()
                - started
            ) * 1000

            return {
                "success": True,
                "latency_ms":
                    latency_ms,
                "result":
                    data.get(
                        "result"
                    ),
            }

        if status == "failed":

            return {
                "success": False,
                "error":
                    data.get(
                        "error"
                    ),
            }

        await asyncio.sleep(
            0.02
        )


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:8000"
        ),
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    image_bytes = (
        image_path.read_bytes()
    )

    content_type = (
        mimetypes.guess_type(
            image_path.name
        )[0]
        or "image/jpeg"
    )

    semaphore = (
        asyncio.Semaphore(
            args.concurrency
        )
    )

    async with httpx.AsyncClient() as client:

        print(
            f"Submitting "
            f"{args.requests} jobs..."
        )

        benchmark_start = (
            perf_counter()
        )

        submission_tasks = [
            submit_job(
                client,
                semaphore,
                args.url,
                image_bytes,
                image_path.name,
                content_type,
            )
            for _ in range(
                args.requests
            )
        ]

        job_ids = await asyncio.gather(
            *submission_tasks
        )

        print(
            "Waiting for inference "
            "results..."
        )
        result_tasks = [
            wait_for_job(
                client,
                args.url,
                job["job_id"],
                job["started"],
            )
            for job in job_ids
        ]

        results = await asyncio.gather(
            *result_tasks
        )

    total_seconds = (
        perf_counter()
        - benchmark_start
    )

    successful = [
        result
        for result in results
        if result["success"]
    ]

    failed = (
        len(results)
        - len(successful)
    )

    latencies = [
        result["latency_ms"]
        for result in successful
    ]

    observed_batch_sizes = [
        result["result"].get(
            "batch_size",
            1,
        )
        for result in successful
        if result.get("result")
    ]

    throughput = (
        len(successful)
        / total_seconds
    )

    print()
    print(
        "=============================="
    )
    print(
        " ASYNC QUEUE BENCHMARK"
    )
    print(
        "=============================="
    )

    print(
        f"Requests:     "
        f"{args.requests}"
    )

    print(
        f"Successful:   "
        f"{len(successful)}"
    )

    print(
        f"Failed:       "
        f"{failed}"
    )

    print(
        f"Total time:   "
        f"{total_seconds:.3f} sec"
    )

    print(
        f"Throughput:   "
        f"{throughput:.3f} jobs/sec"
    )

    if latencies:

        print(
            f"Average:      "
            f"{statistics.mean(latencies):.3f} ms"
        )

        print(
            f"P50:          "
            f"{percentile(latencies, 0.50):.3f} ms"
        )

        print(
            f"P95:          "
            f"{percentile(latencies, 0.95):.3f} ms"
        )

        print(
            f"P99:          "
            f"{percentile(latencies, 0.99):.3f} ms"
        )

    if observed_batch_sizes:

        print(
            f"Average batch:"
            f" "
            f"{statistics.mean(observed_batch_sizes):.2f}"
        )

        print(
            f"Max batch:    "
            f"{max(observed_batch_sizes)}"
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )