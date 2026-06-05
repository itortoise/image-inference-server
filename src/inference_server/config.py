"""配置系统 - Pydantic 模型定义。"""

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """服务级配置。"""
    http_port: int = 8000
    grpc_port: int = 8001
    max_concurrent_requests: int = 100


class SchedulerConfig(BaseModel):
    """调度器配置。"""
    max_batch_size: int = Field(default=32, ge=1)
    max_wait_ms: float = Field(default=5.0, ge=0)
    queue_capacity: int = Field(default=1000, ge=1)


class TensorConfig(BaseModel):
    """输入/输出张量配置。"""
    name: str
    data_type: str
    dims: list[int]


class PreprocessConfig(BaseModel):
    """预处理配置。"""
    resize: list[int] = Field(default=[224, 224])
    mean: list[float] = Field(default=[0.485, 0.456, 0.406])
    std: list[float] = Field(default=[0.229, 0.224, 0.225])
    pixel_format: str = Field(default="RGB")


class DynamicBatchingConfig(BaseModel):
    """动态批处理配置。"""
    max_batch_size: int = Field(default=32, ge=1)
    max_wait_ms: float = Field(default=5.0, ge=0)


class ModelConfig(BaseModel):
    """模型级配置。"""
    name: str
    backend: str
    max_batch_size: int = Field(default=32, ge=1)
    input: list[TensorConfig]
    output: list[TensorConfig]
    preprocess: PreprocessConfig
    backend_config: dict = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    """指标配置。"""
    enabled: bool = True
    port: int = 8080


class ModelsConfig(BaseModel):
    """模型仓库配置。"""
    model_dir: str = "./models"
    preload: list[str] = Field(default_factory=list)


class ServiceConfig(BaseModel):
    """完整服务配置。"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
