# AI Inference Platform

A production-oriented ML serving project built incrementally with FastAPI and PyTorch.

## Production AI Inference Platform

A containerized machine-learning inference platform that serves
image-classification models through a FastAPI API and asynchronous
Redis-backed worker queue.

### Highlights

- PyTorch and ONNX Runtime inference backends
- GPU inference with NVIDIA CUDA
- Dynamic request batching
- Redis asynchronous job queue
- PostgreSQL model registry
- FastAPI REST API
- Prometheus metrics
- Grafana observability
- React operations dashboard
- Docker Compose deployment
- Concurrent load and latency benchmarking

### Architecture

React Dashboard
        |
        v
     FastAPI
        |
        v
      Redis
        |
        v
Inference Worker
   /          \
PyTorch      ONNX Runtime
        |
        v
     NVIDIA GPU

PostgreSQL stores model metadata and versions.

Prometheus collects API, worker, Redis, and PostgreSQL metrics.
Grafana provides monitoring dashboards.

### Performance

Tests used 100 asynchronous inference jobs at concurrency 20.

| Backend | Batch | Throughput | Avg Latency | P95 |
|---|---:|---:|---:|---:|
| PyTorch CUDA | 4 | ~6.87 jobs/s | ~8.14 s | ~13.97 s |
| ONNX Runtime CUDA | 4 | ~6.14 jobs/s | ~9.36 s | ~15.93 s |

For this workload, PyTorch CUDA produced the best throughput/latency
tradeoff and remains the default inference backend.

### Run

```bash
docker compose up --build -d
```


Open:

- Dashboard: http://localhost:5173
- API docs: http://localhost:8000/docs
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090