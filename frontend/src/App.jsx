import { useCallback, useEffect, useState } from "react";

const PYTORCH_RESULTS = {
  throughput: 6.87,
  avgLatency: 8.14,
  p95: 13.97,
};

const ONNX_RESULTS = {
  throughput: 6.14,
  avgLatency: 9.36,
  p95: 15.93,
};

function getMetric(metrics, names, fallback = "—") {
  for (const name of names) {
    if (metrics?.[name] !== undefined && metrics?.[name] !== null) {
      return metrics[name];
    }
  }

  return fallback;
}

function formatNumber(value, digits = 2) {
  if (typeof value !== "number") {
    return value;
  }

  return value.toFixed(digits);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [queue, setQueue] = useState({});
  const [metrics, setMetrics] = useState({});
  const [models, setModels] = useState([]);

  const [file, setFile] = useState(null);
  const [topK, setTopK] = useState(5);
  const [job, setJob] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const refreshDashboard = useCallback(async () => {
    try {
      const [
        healthResponse,
        queueResponse,
        metricsResponse,
        modelsResponse,
      ] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/queue/health"),
        fetch("/api/metrics"),
        fetch("/api/models"),
      ]);

      if (!healthResponse.ok) {
        throw new Error("API unavailable");
      }

      setApiOnline(true);

      if (queueResponse.ok) {
        setQueue(await queueResponse.json());
      }

      if (metricsResponse.ok) {
        const data = await metricsResponse.json();
        setMetrics(data.api ?? {});
      }

      if (modelsResponse.ok) {
        setModels(await modelsResponse.json());
      }
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    refreshDashboard();

    const timer = setInterval(
      refreshDashboard,
      3000,
    );

    return () => clearInterval(timer);
  }, [refreshDashboard]);

  async function submitJob(event) {
    event.preventDefault();

    if (!file) {
      return;
    }

    setSubmitting(true);
    setJob(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `/api/jobs?top_k=${topK}`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(
          `Job submission failed: ${response.status}`,
        );
      }

      const queuedJob = await response.json();

      setJob(queuedJob);

      for (let attempt = 0; attempt < 120; attempt += 1) {
        await sleep(500);

        const statusResponse = await fetch(
          `/api/jobs/${queuedJob.job_id}`,
        );

        if (!statusResponse.ok) {
          throw new Error("Could not read job status.");
        }

        const currentJob = await statusResponse.json();

        setJob(currentJob);

        if (
          currentJob.status === "completed" ||
          currentJob.status === "failed"
        ) {
          break;
        }
      }
    } catch (error) {
      setJob({
        status: "failed",
        error: error.message,
      });
    } finally {
      setSubmitting(false);
      refreshDashboard();
    }
  }

  const totalRequests = getMetric(
    metrics,
    [
      "total_requests",
      "requests_total",
    ],
    0,
  );

  const errors = getMetric(
    metrics,
    [
      "total_errors",
      "errors",
    ],
    0,
  );

  const p95 = getMetric(
    metrics,
    [
      "p95_latency_ms",
      "p95_ms",
      "p95",
    ],
  );

  const requestRate = getMetric(
    metrics,
    [
      "average_requests_per_second",
      "avg_requests_per_second",
      "requests_per_second",
      "avg_rps",
    ],
  );

  const predictions =
    job?.result?.predictions ?? [];

  const activeModels = models.filter(
    (model) => model.is_active,
  ).length;

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="eyebrow">
            PRODUCTION ML SYSTEM
          </div>

          <h1>
            AI Inference Platform
          </h1>

          <p>
            Asynchronous model serving, dynamic batching,
            observability, and multi-backend inference.
          </p>
        </div>

        <div className="header-actions">
          <a
            href="http://127.0.0.1:3000"
            target="_blank"
          >
            Grafana
          </a>

          <a
            href="http://127.0.0.1:9090/targets"
            target="_blank"
          >
            Prometheus
          </a>

          <a
            href="/api/docs"
            target="_blank"
          >
            API Docs
          </a>
        </div>
      </header>

      <main>
        <section className="status-grid">
          <StatusCard
            label="API"
            value={apiOnline ? "Online" : "Offline"}
            good={apiOnline}
          />

          <StatusCard
            label="Redis"
            value={
              queue.redis_connected
                ? "Connected"
                : "Disconnected"
            }
            good={queue.redis_connected}
          />

          <StatusCard
            label="Queue Depth"
            value={queue.queue_size ?? "—"}
          />

          <StatusCard
            label="Requests"
            value={totalRequests}
          />

          <StatusCard
            label="Errors"
            value={errors}
            good={errors === 0}
          />

          <StatusCard
            label="P95 API Latency"
            value={
              typeof p95 === "number"
                ? `${formatNumber(p95)} ms`
                : p95
            }
          />

          <StatusCard
            label="Request Rate"
            value={
              typeof requestRate === "number"
                ? `${formatNumber(requestRate)} /s`
                : requestRate
            }
          />

          <StatusCard
            label="Active Models"
            value={activeModels}
          />
        </section>

        <div className="two-column">
          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-number">
                  01
                </span>

                <h2>
                  Run Inference
                </h2>
              </div>

              <span className="badge">
                Redis Queue
              </span>
            </div>

            <form
              className="inference-form"
              onSubmit={submitJob}
            >
              <label className="file-picker">
                <span>
                  Select an image
                </span>

                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) =>
                    setFile(
                      event.target.files?.[0] ?? null,
                    )
                  }
                />

                <strong>
                  {file
                    ? file.name
                    : "Choose JPG, PNG, or WEBP"}
                </strong>
              </label>

              <div className="form-row">
                <label>
                  Top K

                  <select
                    value={topK}
                    onChange={(event) =>
                      setTopK(
                        Number(event.target.value),
                      )
                    }
                  >
                    {[1, 3, 5, 10].map((value) => (
                      <option
                        key={value}
                        value={value}
                      >
                        {value}
                      </option>
                    ))}
                  </select>
                </label>

                <button
                  disabled={!file || submitting}
                >
                  {submitting
                    ? "Processing..."
                    : "Submit Job"}
                </button>
              </div>
            </form>

            {job && (
              <div className="job-result">
                <div className="job-header">
                  <span>
                    Status
                  </span>

                  <strong
                    className={`status-${job.status}`}
                  >
                    {job.status}
                  </strong>
                </div>

                {job.job_id && (
                  <code>
                    {job.job_id}
                  </code>
                )}

                {job.result && (
                  <>
                    <div className="result-meta">
                      <span>
                        Batch
                        <strong>
                          {job.result.batch_size}
                        </strong>
                      </span>

                      <span>
                        Inference
                        <strong>
                          {formatNumber(
                            job.result.inference_ms,
                          )}{" "}
                          ms
                        </strong>
                      </span>

                      <span>
                        Device
                        <strong>
                          {job.device ?? "worker"}
                        </strong>
                      </span>
                    </div>

                    <div className="predictions">
                      {predictions.map(
                        (prediction, index) => (
                          <div
                            className="prediction"
                            key={`${prediction.class_id}-${index}`}
                          >
                            <span>
                              {index + 1}
                            </span>

                            <div>
                              <strong>
                                {prediction.label}
                              </strong>

                              <small>
                                Class{" "}
                                {prediction.class_id}
                              </small>
                            </div>

                            <b>
                              {(
                                prediction.probability *
                                100
                              ).toFixed(2)}
                              %
                            </b>
                          </div>
                        ),
                      )}
                    </div>
                  </>
                )}

                {job.error && (
                  <div className="error">
                    {job.error}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-number">
                  02
                </span>

                <h2>
                  Model Registry
                </h2>
              </div>

              <span className="badge">
                PostgreSQL
              </span>
            </div>

            {models.length === 0 ? (
              <div className="empty">
                No models registered yet.
              </div>
            ) : (
              <div className="model-list">
                {models.slice(0, 8).map((model) => (
                  <div
                    className="model-row"
                    key={model.id}
                  >
                    <div>
                      <strong>
                        {model.name}
                      </strong>

                      <small>
                        {model.framework} · {model.task}
                      </small>
                    </div>

                    <div className="model-version">
                      v{model.version}

                      {model.is_active && (
                        <span>
                          ACTIVE
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>

        <section className="panel benchmark-panel">
          <div className="panel-heading">
            <div>
              <span className="section-number">
                03
              </span>

              <h2>
                Backend Benchmark
              </h2>
            </div>

            <span className="badge">
              Batch 4 · Concurrency 20
            </span>
          </div>

          <div className="benchmark-grid">
            <BenchmarkCard
              name="PyTorch CUDA"
              winner
              data={PYTORCH_RESULTS}
            />

            <BenchmarkCard
              name="ONNX Runtime CUDA"
              data={ONNX_RESULTS}
            />
          </div>

          <p className="benchmark-note">
            Measured locally using 100 queued inference jobs.
            PyTorch remains the default backend because it
            delivered higher throughput and lower latency in
            this workload.
          </p>
        </section>

        <section className="architecture">
          <span className="section-number">
            04
          </span>

          <h2>
            System Architecture
          </h2>

          <div className="architecture-flow">
            <ArchitectureNode text="React" />
            <Arrow />
            <ArchitectureNode text="FastAPI" />
            <Arrow />
            <ArchitectureNode text="Redis" />
            <Arrow />
            <ArchitectureNode text="Worker" />
            <Arrow />
            <ArchitectureNode text="PyTorch / ONNX" />
          </div>

          <div className="architecture-secondary">
            PostgreSQL model registry · Prometheus metrics ·
            Grafana dashboards · Docker Compose
          </div>
        </section>
      </main>

      <footer>
        Production AI Inference Platform · v1.0
      </footer>
    </div>
  );
}

function StatusCard({
  label,
  value,
  good,
}) {
  return (
    <div className="status-card">
      <div className="status-label">
        {label}

        {good !== undefined && (
          <span
            className={
              good
                ? "status-dot good"
                : "status-dot bad"
            }
          />
        )}
      </div>

      <strong>
        {value}
      </strong>
    </div>
  );
}

function BenchmarkCard({
  name,
  data,
  winner = false,
}) {
  return (
    <article
      className={`benchmark-card ${
        winner ? "winner" : ""
      }`}
    >
      <div className="benchmark-title">
        <h3>
          {name}
        </h3>

        {winner && (
          <span>
            DEFAULT
          </span>
        )}
      </div>

      <div className="benchmark-values">
        <div>
          <span>
            Throughput
          </span>

          <strong>
            {data.throughput}
          </strong>

          <small>
            jobs/s
          </small>
        </div>

        <div>
          <span>
            Avg latency
          </span>

          <strong>
            {data.avgLatency}
          </strong>

          <small>
            sec
          </small>
        </div>

        <div>
          <span>
            P95
          </span>

          <strong>
            {data.p95}
          </strong>

          <small>
            sec
          </small>
        </div>
      </div>
    </article>
  );
}

function ArchitectureNode({
  text,
}) {
  return (
    <div className="architecture-node">
      {text}
    </div>
  );
}

function Arrow() {
  return (
    <span className="arrow">
      →
    </span>
  );
}