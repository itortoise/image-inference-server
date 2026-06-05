import pytest
from pydantic import ValidationError

from inference_server.config import ServerConfig, SchedulerConfig, ModelConfig, ServiceConfig


class TestServerConfig:
    def test_default_values(self):
        config = ServerConfig()
        assert config.http_port == 8000
        assert config.grpc_port == 8001
        assert config.max_concurrent_requests == 100

    def test_custom_values(self):
        config = ServerConfig(http_port=9000, grpc_port=9001)
        assert config.http_port == 9000
        assert config.grpc_port == 9001


class TestSchedulerConfig:
    def test_valid_config(self):
        config = SchedulerConfig(max_batch_size=32, max_wait_ms=5.0, queue_capacity=1000)
        assert config.max_batch_size == 32

    def test_invalid_max_batch_size(self):
        with pytest.raises(ValidationError):
            SchedulerConfig(max_batch_size=0, max_wait_ms=5.0, queue_capacity=1000)

    def test_invalid_queue_capacity(self):
        with pytest.raises(ValidationError):
            SchedulerConfig(max_batch_size=32, max_wait_ms=5.0, queue_capacity=0)


class TestModelConfig:
    def test_valid_config(self):
        config = ModelConfig(
            name="resnet50",
            backend="onnxruntime",
            max_batch_size=32,
            input=[{"name": "images", "data_type": "FP32", "dims": [3, 224, 224]}],
            output=[{"name": "output", "data_type": "FP32", "dims": [1000]}],
            preprocess={"resize": [224, 224], "mean": [0.485, 0.456, 0.406]},
        )
        assert config.name == "resnet50"
        assert config.backend == "onnxruntime"
