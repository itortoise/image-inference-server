"""Prometheus 指标暴露。"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server


class InferenceMetrics:
    """推理服务指标。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

        # 请求计数
        self.request_total = Counter(
            "inference_request_total",
            "Total inference requests",
            ["model", "status"],
        )

        # 推理次数（一个 batch 为 N 算 N 次）
        self.inference_count = Counter(
            "inference_count_total",
            "Total inferences performed",
            ["model"],
        )

        # batch 执行次数
        self.exec_count = Counter(
            "inference_exec_count_total",
            "Total batch executions",
            ["model"],
        )

        # 队列深度
        self.pending_requests = Gauge(
            "pending_request_count",
            "Pending requests in queue",
            ["model"],
        )

        # 分段延迟
        self.queue_duration = Histogram(
            "queue_duration_seconds",
            "Time spent waiting in queue",
            ["model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.infer_duration = Histogram(
            "compute_infer_duration_seconds",
            "Inference computation time",
            ["model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        # batch 大小分布
        self.batch_size = Histogram(
            "batch_size_histogram",
            "Batch size distribution",
            ["model"],
            buckets=[1, 2, 4, 8, 16, 32, 64],
        )

    def record_request(self, status: str = "success"):
        self.request_total.labels(model=self.model_name, status=status).inc()

    def record_inference(self, count: int):
        self.inference_count.labels(model=self.model_name).inc(count)

    def record_execution(self):
        self.exec_count.labels(model=self.model_name).inc()

    def set_pending(self, count: int):
        self.pending_requests.labels(model=self.model_name).set(count)

    def observe_queue_duration(self, duration: float):
        self.queue_duration.labels(model=self.model_name).observe(duration)

    def observe_infer_duration(self, duration: float):
        self.infer_duration.labels(model=self.model_name).observe(duration)

    def observe_batch_size(self, size: int):
        self.batch_size.labels(model=self.model_name).observe(size)
