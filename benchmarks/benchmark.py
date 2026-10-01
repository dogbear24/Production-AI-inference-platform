import argparse
import asyncio
import csv
import mimetypes
import statistics
from pathlib import Path
from time import perf_counter

import httpx


def percentile(values, percent):
    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


async def send_request(
    client,
    url,
    image_bytes,
    filename,
    content_type,
    semaphore,
):
    async with semaphore:

        start = perf_counter()

        try:
            response = await client.post(
                url,
                files={
                    "file": (
                        filename,
                        image_bytes,
                        content_type,
                    )
                },
                timeout=60.0,
            )

            latency_ms = (
                perf_counter() - start
            ) * 1000

            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }

        except Exception as exc:

            latency_ms = (
                perf_counter() - start
            ) * 1000

            return {
                "success": False,
                "status_code": None,
                "latency_ms": latency_ms,
                "error": str(exc),
            }


async def run_benchmark(
    url,
    image_path,
    request_count,
    concurrency,
    warmup,
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    content_type = (
        mimetypes.guess_type(image_path.name)[0]
        or "image/jpeg"
    )

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        # -------------------------
        # WARMUP
        # -------------------------

        print(
            f"\nWarming up with {warmup} requests..."
        )

        for _ in range(warmup):
            await send_request(
                client,
                url,
                image_bytes,
                image_path.name,
                content_type,
                semaphore,
            )

        # -------------------------
        # BENCHMARK
        # -------------------------

        print(
            f"Running {request_count} requests "
            f"with concurrency={concurrency}..."
        )

        benchmark_start = perf_counter()

        tasks = [
            send_request(
                client,
                url,
                image_bytes,
                image_path.name,
                content_type,
                semaphore,
            )
            for _ in range(request_count)
        ]

        results = await asyncio.gather(*tasks)

        total_seconds = (
            perf_counter() - benchmark_start
        )

    latencies = [
        result["latency_ms"]
        for result in results
        if result["success"]
    ]

    successes = sum(
        result["success"]
        for result in results
    )

    failures = request_count - successes

    throughput = (
        successes / total_seconds
        if total_seconds > 0
        else 0
    )

    average_latency = (
        statistics.mean(latencies)
        if latencies
        else 0
    )

    result = {
        "requests": request_count,
        "concurrency": concurrency,
        "successful": successes,
        "failed": failures,
        "total_time_seconds": round(
            total_seconds,
            3,
        ),
        "throughput_requests_per_second": round(
            throughput,
            3,
        ),
        "average_latency_ms": round(
            average_latency,
            3,
        ),
        "p50_latency_ms": round(
            percentile(latencies, 0.50),
            3,
        ),
        "p95_latency_ms": round(
            percentile(latencies, 0.95),
            3,
        ),
        "p99_latency_ms": round(
            percentile(latencies, 0.99),
            3,
        ),
    }

    return result


def print_results(result):

    print("\n==============================")
    print("      BENCHMARK RESULTS")
    print("==============================")

    print(
        f"Requests:      "
        f"{result['requests']}"
    )

    print(
        f"Concurrency:   "
        f"{result['concurrency']}"
    )

    print(
        f"Successful:    "
        f"{result['successful']}"
    )

    print(
        f"Failed:        "
        f"{result['failed']}"
    )

    print(
        f"Total time:    "
        f"{result['total_time_seconds']} sec"
    )

    print(
        f"Throughput:    "
        f"{result['throughput_requests_per_second']} req/sec"
    )

    print(
        f"Average:       "
        f"{result['average_latency_ms']} ms"
    )

    print(
        f"P50:           "
        f"{result['p50_latency_ms']} ms"
    )

    print(
        f"P95:           "
        f"{result['p95_latency_ms']} ms"
    )

    print(
        f"P99:           "
        f"{result['p99_latency_ms']} ms"
    )


def save_result(result, output_file):

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = output_path.exists()

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


async def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the AI inference API."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to an image to upload.",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of benchmark requests.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum simultaneous requests.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup requests before measurement.",
    )

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:8000/predict"
        ),
        help="Prediction endpoint.",
    )

    parser.add_argument(
        "--output",
        default=(
            "benchmarks/results.csv"
        ),
        help="CSV output file.",
    )

    args = parser.parse_args()

    result = await run_benchmark(
        url=args.url,
        image_path=args.image,
        request_count=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
    )

    print_results(result)

    save_result(
        result,
        args.output,
    )

    print(
        f"\nSaved result to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    asyncio.run(main())