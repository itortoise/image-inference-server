# 昇腾图像推理服务实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个支持动态批处理的通用图像识别推理服务，Backend 与 Serving 解耦，支持 HTTP/gRPC 双协议，可运行在昇腾 NPU 上。

**Architecture:** 纯 Python 异步架构。asyncio 调度器收集请求凑 batch，ThreadPool 并行预处理，Backend 抽象基类支持 ONNX Runtime（含 CANN EP）等推理引擎。FastAPI 提供 HTTP 服务，grpcio 提供 gRPC 服务。

**Tech Stack:** Python 3.10+, FastAPI, grpcio, onnxruntime, numpy, opencv-python, pydantic, pyyaml, prometheus-client, pytest, ruff, mypy

---

## 文件结构映射

```
inference-server/
├── pyproject.toml                          # Task 1
├── Dockerfile                              # Task 13
├── configs/
│   └── service.yaml                        # Task 13
├── models/
│   └── resnet50/
│       ├── 1/
│       │   └── model.onnx                  # Task 13 (提供测试模型)
│       └── config.yaml                     # Task 13
├── src/
│   └── inference_server/
│       ├── __init__.py                     # Task 1
│       ├── main.py                         # Task 12
│       ├── config.py                       # Task 4
│       ├── api/
│       │   ├── __init__.py                 # Task 1
│       │   ├── http_server.py              # Task 8
│       │   ├── grpc_server.py              # Task 9
│       │   ├── schemas.py                  # Task 8
│       │   └── inference.proto             # Task 9
│       ├── core/
│       │   ├── __init__.py                 # Task 1
│       │   ├── scheduler.py                # Task 6
│       │   ├── preprocessor.py             # Task 5
│       │   ├── postprocessor.py            # Task 7
│       │   └── request.py                  # Task 4
│       ├── backend/
│       │   ├── __init__.py                 # Task 2
│       │   ├── base.py                     # Task 2
│       │   ├── onnx_backend.py             # Task 3
│       │   └── registry.py                 # Task 2
│       ├── model_manager.py                # Task 11
│       └── metrics.py                      # Task 10
└── tests/
    ├── __init__.py                         # Task 1
    ├── test_scheduler.py                   # Task 6
    ├── test_backend.py                     # Task 3
    ├── test_preprocessor.py                # Task 5
    └── test_integration.py                 # Task 14
```

---

## Task 1: 项目基础结构

**目标:** 创建项目目录结构、pyproject.toml 和空文件。

**Files:**
- Create: `pyproject.toml`
- Create: `src/inference_server/__init__.py`
- Create: `src/inference_server/api/__init__.py`
- Create: `src/inference_server/core/__init__.py`
- Create: `src/inference_server/backend/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
cd /home/macbook/workspace/personal_project/myzone/inference-server
mkdir -p src/inference_server/{api,core,backend}
mkdir -p tests
mkdir -p configs
mkdir -p models/resnet50/1
```

- [ ] **Step 2: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "inference-server"
version = "0.1.0"
description = "通用图像识别推理服务"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "grpcio>=1.56",
    "grpcio-tools>=1.56",
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "pillow>=10.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "prometheus-client>=0.17",
    "python-multipart>=0.0.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.1",
    "httpx>=0.24",
    "ruff>=0.1",
    "mypy>=1.5",
]
prod = [
    "gunicorn>=21.0",
]

[project.scripts]
inference-server = "inference_server.main:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
```

- [ ] **Step 3: 创建空 __init__.py 文件**

```bash
touch src/inference_server/__init__.py
touch src/inference_server/api/__init__.py
touch src/inference_server/core/__init__.py
touch src/inference_server/backend/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure with pyproject.toml"
```

---

## Task 2: Backend 抽象基类 + 注册表

**目标:** 实现 Backend 抽象基类和装饰器注册机制。

**Files:**
- Create: `src/inference_server/backend/base.py`
- Create: `src/inference_server/backend/registry.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: 写测试 - Backend 基类不能被直接实例化**

Create `tests/test_backend.py`:

```python
import pytest
from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend, get_backend


class TestBackendBase:
    def test_backend_is_abstract(self):
        """Backend 基类不能直接实例化"""
        with pytest.raises(TypeError):
            Backend()


class TestBackendRegistry:
    def test_register_and_get_backend(self):
        """测试注册和获取 backend"""

        @register_backend("test_backend")
        class TestBackend(Backend):
            def initialize(self, model_path, config):
                pass

            def infer(self, inputs):
                return inputs

            def get_input_specs(self):
                return []

            def get_output_specs(self):
                return []

            def destroy(self):
                pass

        cls = get_backend("test_backend")
        assert cls.__name__ == "TestBackend"

    def test_get_unknown_backend_raises(self):
        """获取未注册的 backend 应该报错"""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/macbook/workspace/personal_project/myzone/inference-server
pytest tests/test_backend.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` (文件不存在)

- [ ] **Step 3: 实现 Backend 基类**

Create `src/inference_server/backend/base.py`:

```python
"""Backend 抽象基类 - 所有推理后端必须继承此类。"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Backend(ABC):
    """推理后端抽象基类。

    Backend 只负责"张量进张量出"，不接触 HTTP/gRPC 或原始图像数据。
    """

    @abstractmethod
    def initialize(self, model_path: str, config: Dict) -> None:
        """加载模型，准备推理环境。

        Args:
            model_path: 模型文件路径
            config: backend 特定配置（从 model config.yaml 的 backend_config 透传）
        """
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理。

        Args:
            inputs: {input_name: tensor}，张量已预处理好，shape 为 [B, ...]

        Returns:
            {output_name: tensor}，shape 为 [B, ...]
        """
        pass

    @abstractmethod
    def get_input_specs(self) -> list[dict]:
        """返回输入张量规格（name, dtype, shape）。"""
        pass

    @abstractmethod
    def get_output_specs(self) -> list[dict]:
        """返回输出张量规格。"""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """释放资源。"""
        pass
```

- [ ] **Step 4: 实现注册表**

Create `src/inference_server/backend/registry.py`:

```python
"""Backend 注册表 - 通过装饰器注册后端实现。"""

from typing import Dict, Type

from inference_server.backend.base import Backend

_BACKENDS: Dict[str, Type[Backend]] = {}


def register_backend(name: str):
    """装饰器，注册 backend 实现。

    Usage:
        @register_backend("onnxruntime")
        class ONNXRuntimeBackend(Backend):
            ...
    """

    def wrapper(cls: Type[Backend]) -> Type[Backend]:
        if not issubclass(cls, Backend):
            raise TypeError(f"Backend {cls.__name__} must inherit from Backend")
        _BACKENDS[name] = cls
        return cls

    return wrapper


def get_backend(name: str) -> Type[Backend]:
    """根据名称获取已注册的 backend 类。"""
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    """返回所有已注册的 backend 名称。"""
    return list(_BACKENDS.keys())
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_backend.py -v
```

Expected: 3 tests passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add Backend abstract base class and registry"
```

---

## Task 3: ONNX Runtime Backend 实现

**目标:** 实现 ONNXRuntimeBackend，支持通过 Execution Provider 选择硬件。

**Files:**
- Create: `src/inference_server/backend/onnx_backend.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: 写测试 - ONNX Backend 初始化和推理**

Append to `tests/test_backend.py`:

```python
import numpy as np
import tempfile
import os


class TestONNXRuntimeBackend:
    def _create_test_onnx_model(self) -> str:
        """创建一个简单的 ONNX 模型用于测试：y = x * 2 + 1"""
        try:
            import onnx
            from onnx import numpy_helper, TensorProto
            from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
        except ImportError:
            pytest.skip("onnx package not installed")

        # 输入: x [B, 3, 224, 224]
        # 输出: y [B, 3, 224, 224] = x * 2 + 1
        input_info = make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 224, 224])
        output_info = make_tensor_value_info("y", TensorProto.FLOAT, [None, 3, 224, 224])

        # 创建常量: scale=2, bias=1
        scale = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="scale")
        bias = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bias")

        node_mul = make_node("Mul", ["x", "scale"], ["mul_out"])
        node_add = make_node("Add", ["mul_out", "bias"], ["y"])

        graph = make_graph(
            [node_mul, node_add],
            "test_model",
            [input_info],
            [output_info],
            initializer=[scale, bias],
        )
        model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        # 保存到临时文件
        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.write(fd, onnx._serialize(model))
        os.close(fd)
        return path

    def test_onnx_backend_initialize_and_infer(self):
        """测试 ONNX Backend 初始化和推理"""
        from inference_server.backend.onnx_backend import ONNXRuntimeBackend

        model_path = self._create_test_onnx_model()
        backend = ONNXRuntimeBackend()
        backend.initialize(model_path, {"providers": ["CPUExecutionProvider"]})

        # 检查输入输出规格
        input_specs = backend.get_input_specs()
        assert len(input_specs) == 1
        assert input_specs[0]["name"] == "x"

        output_specs = backend.get_output_specs()
        assert len(output_specs) == 1
        assert output_specs[0]["name"] == "y"

        # 推理
        x = np.ones((2, 3, 224, 224), dtype=np.float32)
        outputs = backend.infer({"x": x})

        assert "y" in outputs
        expected = x * 2 + 1
        np.testing.assert_allclose(outputs["y"], expected, rtol=1e-5)

        backend.destroy()
        os.unlink(model_path)

    def test_onnx_backend_providers_config(self):
        """测试 providers 配置解析"""
        from inference_server.backend.onnx_backend import ONNXRuntimeBackend

        model_path = self._create_test_onnx_model()
        backend = ONNXRuntimeBackend()

        # 带 options 的 provider 配置
        config = {
            "providers": [
                {"name": "CPUExecutionProvider", "options": {}},
            ]
        }
        backend.initialize(model_path, config)
        assert backend.session is not None

        backend.destroy()
        os.unlink(model_path)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_backend.py::TestONNXRuntimeBackend -v
```

Expected: ImportError (onnx_backend 不存在)

- [ ] **Step 3: 实现 ONNXRuntimeBackend**

Create `src/inference_server/backend/onnx_backend.py`:

```python
"""ONNX Runtime Backend 实现。"""

from typing import Dict

import numpy as np
import onnxruntime as ort

from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend


@register_backend("onnxruntime")
class ONNXRuntimeBackend(Backend):
    """ONNX Runtime 推理后端。

    通过 providers 配置支持不同硬件：
    - CANNExecutionProvider: 昇腾 NPU
    - CUDAExecutionProvider: NVIDIA GPU
    - CPUExecutionProvider: CPU (兜底)
    """

    def initialize(self, model_path: str, config: Dict) -> None:
        """加载 ONNX 模型。

        Args:
            model_path: ONNX 模型文件路径
            config: 必须包含 providers 列表
        """
        providers_config = config.get("providers", [
            {"name": "CPUExecutionProvider"}
        ])

        # 解析 providers 配置
        providers = []
        for p in providers_config:
            if isinstance(p, dict):
                providers.append((p["name"], p.get("options", {})))
            elif isinstance(p, str):
                providers.append(p)
            else:
                raise ValueError(f"Invalid provider config: {p}")

        self.session = ort.InferenceSession(model_path, providers=providers)

        # 缓存输入输出规格
        self._input_specs = self._extract_specs(self.session.get_inputs())
        self._output_specs = self._extract_specs(self.session.get_outputs())

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行 ONNX 推理。"""
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(output_names, inputs)
        return dict(zip(output_names, outputs))

    def get_input_specs(self) -> list[dict]:
        return self._input_specs

    def get_output_specs(self) -> list[dict]:
        return self._output_specs

    def destroy(self) -> None:
        del self.session
        self.session = None

    @staticmethod
    def _extract_specs(io_info) -> list[dict]:
        """从 ONNX session 的输入/输出信息中提取规格。"""
        specs = []
        for info in io_info:
            shape = []
            for dim in info.shape:
                if isinstance(dim, (int, float)):
                    shape.append(int(dim) if dim > 0 else -1)
                else:
                    shape.append(-1)
            specs.append({
                "name": info.name,
                "dtype": info.type,
                "shape": shape,
            })
        return specs
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_backend.py -v
```

Expected: 5 tests passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add ONNXRuntimeBackend with provider config support"
```

---

## Task 4: 请求数据模型 + 配置系统

**目标:** 实现 InferenceRequest 数据类和 Pydantic 配置模型。

**Files:**
- Create: `src/inference_server/core/request.py`
- Create: `src/inference_server/config.py`

- [ ] **Step 1: 写测试 - 配置模型校验**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError

- [ ] **Step 3: 实现请求数据模型**

Create `src/inference_server/core/request.py`:

```python
"""推理请求数据模型。"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class InferenceRequest:
    """单个推理请求。"""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    model_name: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future = field(default=None, repr=False)
    enqueue_time: float = field(default=0.0)

    def set_result(self, result: Any) -> None:
        """设置异步结果。"""
        if self.future and not self.future.done():
            self.future.set_result(result)

    def set_exception(self, exc: Exception) -> None:
        """设置异步异常。"""
        if self.future and not self.future.done():
            self.future.set_exception(exc)
```

- [ ] **Step 4: 实现配置模型**

Create `src/inference_server/config.py`:

```python
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
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_config.py -v
```

Expected: 5 tests passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add request model and pydantic config system"
```

---

## Task 5: 图像预处理

**目标:** 实现图像预处理，支持 resize、normalize、format 转换。

**Files:**
- Create: `src/inference_server/core/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: 写测试 - 预处理流程**

Create `tests/test_preprocessor.py`:

```python
import numpy as np
import pytest

from inference_server.config import PreprocessConfig
from inference_server.core.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    def test_preprocess_rgb_image(self):
        """测试 RGB 图像预处理"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            pixel_format="RGB",
        )
        preprocessor = ImagePreprocessor(config)

        # 模拟 RGB 图像: 400x300x3
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        result = preprocessor.process(image)

        assert result.shape == (3, 224, 224)  # NCHW
        assert result.dtype == np.float32

    def test_preprocess_with_normalization(self):
        """测试包含归一化的预处理"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            pixel_format="RGB",
        )
        preprocessor = ImagePreprocessor(config)

        # 全白图像 (255, 255, 255)
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        result = preprocessor.process(image)

        # 归一化后: (255/255 - 0.5) / 0.5 = 1.0
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_preprocess_bgr_to_rgb(self):
        """测试 BGR 转 RGB"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            pixel_format="BGR",
        )
        preprocessor = ImagePreprocessor(config)

        # BGR 图像: 蓝色通道=255
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # B = 255

        result = preprocessor.process(image)

        # 转换后 RGB: R=0, G=0, B=255/255=1.0
        assert result[0, 0, 0] == 0.0  # R
        assert result[1, 0, 0] == 0.0  # G
        assert result[2, 0, 0] == 1.0  # B

    def test_preprocess_batch(self):
        """测试批量预处理"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            pixel_format="RGB",
        )
        preprocessor = ImagePreprocessor(config)

        images = [
            np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),
            np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
        ]

        results = preprocessor.process_batch(images)

        assert len(results) == 2
        assert all(r.shape == (3, 224, 224) for r in results)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: ImportError

- [ ] **Step 3: 实现图像预处理器**

Create `src/inference_server/core/preprocessor.py`:

```python
"""图像预处理模块。"""

from typing import List

import cv2
import numpy as np

from inference_server.config import PreprocessConfig


class ImagePreprocessor:
    """图像预处理器。

    执行以下操作：
    1. 格式转换 (BGR -> RGB 等)
    2. Resize
    3. 归一化 (mean/std)
    4. 维度转换 (HWC -> CHW)
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.target_size = tuple(config.resize)
        self.mean = np.array(config.mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(config.std, dtype=np.float32).reshape(3, 1, 1)
        self.pixel_format = config.pixel_format.upper()

    def process(self, image: np.ndarray) -> np.ndarray:
        """预处理单张图像。

        Args:
            image: numpy array, HWC format, dtype uint8

        Returns:
            numpy array, CHW format, dtype float32, normalized
        """
        # 确保是 uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 格式转换
        if self.pixel_format == "RGB" and image.shape[2] == 3:
            # OpenCV 默认读取 BGR，如果输入是 BGR 需要转 RGB
            # 但这里假设输入已经是目标格式，或者由调用方处理
            pass  # 保持原样
        elif self.pixel_format == "BGR" and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 转 float32 并归一化到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # 减去 mean，除以 std
        if self.pixel_format == "RGB":
            # HWC -> CHW
            chw = np.transpose(normalized, (2, 0, 1))
        else:
            chw = np.transpose(normalized, (2, 0, 1))

        # 应用 mean/std
        result = (chw - self.mean) / self.std

        return result

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """批量预处理。"""
        return [self.process(img) for img in images]

    def merge_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        """将预处理后的单张图像张量合并成 batch 张量 [B, C, H, W]。"""
        return np.stack(tensors, axis=0)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: 4 tests passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add image preprocessor with resize, normalize, format conversion"
```

---

## Task 6: 动态批处理调度器

**目标:** 实现核心调度器，凑 batch + 超时触发 + 结果分发。

**Files:**
- Create: `src/inference_server/core/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: 写测试 - 调度器核心逻辑**

Create `tests/test_scheduler.py`:

```python
import asyncio
import time
from unittest.mock import Mock

import numpy as np
import pytest

from inference_server.config import SchedulerConfig
from inference_server.core.request import InferenceRequest
from inference_server.core.scheduler import DynamicBatcher


class TestDynamicBatcher:
    @pytest.fixture
    def mock_backend(self):
        backend = Mock()
        backend.infer = Mock(return_value={"output": np.ones((2, 10))})
        return backend

    @pytest.fixture
    def mock_preprocessor(self):
        preprocessor = Mock()
        preprocessor.process_batch = Mock(return_value=[np.ones((3, 224, 224))])
        preprocessor.merge_batch = Mock(return_value=np.ones((2, 3, 224, 224)))
        return preprocessor

    @pytest.fixture
    def mock_postprocessor(self):
        postprocessor = Mock()
        postprocessor.process = Mock(return_value={"class": "cat", "score": 0.99})
        return postprocessor

    @pytest.fixture
    def scheduler(self):
        return DynamicBatcher(
            max_batch_size=4,
            max_wait_ms=50.0,
            queue_capacity=100,
        )

    @pytest.mark.asyncio
    async def test_submit_and_get_future(self, scheduler):
        """测试提交请求返回 Future"""
        req = InferenceRequest(model_name="test", inputs={"image": b"data"})
        future = await scheduler.submit(req)
        assert isinstance(future, asyncio.Future)
        assert not future.done()

    @pytest.mark.asyncio
    async def test_queue_full_rejects(self, scheduler):
        """测试队列满时拒绝请求"""
        # 填满队列
        for _ in range(100):
            req = InferenceRequest(model_name="test", inputs={})
            await scheduler.submit(req)

        # 第 101 个应该被拒绝
        req = InferenceRequest(model_name="test", inputs={})
        with pytest.raises(Exception):
            await scheduler.submit(req)

    @pytest.mark.asyncio
    async def test_batch_execution(self, scheduler, mock_backend,
                                    mock_preprocessor, mock_postprocessor):
        """测试 batch 推理完整流程"""
        # 启动调度器（在后台运行）
        task = asyncio.create_task(
            scheduler.run(mock_backend, mock_preprocessor, mock_postprocessor)
        )

        # 提交 2 个请求
        req1 = InferenceRequest(model_name="test", inputs={"image": b"data1"})
        req2 = InferenceRequest(model_name="test", inputs={"image": b"data2"})

        future1 = await scheduler.submit(req1)
        future2 = await scheduler.submit(req2)

        # 等待结果（超时触发）
        result1 = await asyncio.wait_for(future1, timeout=1.0)
        result2 = await asyncio.wait_for(future2, timeout=1.0)

        assert result1 == {"class": "cat", "score": 0.99}
        assert result2 == {"class": "cat", "score": 0.99}

        # 验证 backend.infer 被调用，且 batch_size=2
        mock_backend.infer.assert_called_once()
        call_args = mock_backend.infer.call_args[0][0]
        assert call_args["input"].shape[0] == 2  # batch size = 2

        scheduler.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_max_batch_size_trigger(self, scheduler, mock_backend,
                                           mock_preprocessor, mock_postprocessor):
        """测试凑满 max_batch_size 立即触发"""
        scheduler.max_wait_ms = 10000  # 10秒，确保不会超时

        task = asyncio.create_task(
            scheduler.run(mock_backend, mock_preprocessor, mock_postprocessor)
        )

        # 提交 4 个请求（等于 max_batch_size）
        futures = []
        for i in range(4):
            req = InferenceRequest(model_name="test", inputs={"image": f"data{i}"})
            future = await scheduler.submit(req)
            futures.append(future)

        # 应该很快返回（不需要等 10 秒）
        start = time.monotonic()
        results = await asyncio.gather(*[asyncio.wait_for(f, timeout=1.0) for f in futures])
        elapsed = time.monotonic() - start

        assert elapsed < 0.5  # 应该很快，不需要等超时
        assert len(results) == 4

        scheduler.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_scheduler.py -v
```

Expected: ImportError or AttributeError

- [ ] **Step 3: 实现动态批处理调度器**

Create `src/inference_server/core/scheduler.py`:

```python
"""动态批处理调度器。"""

import asyncio
import time
from typing import Any, List

from inference_server.backend.base import Backend
from inference_server.config import SchedulerConfig
from inference_server.core.request import InferenceRequest


class QueueFullError(Exception):
    """队列已满异常。"""
    pass


class DynamicBatcher:
    """动态批处理调度器。

    核心算法：
    1. 首请求到达时触发 batch 窗口
    2. 后续请求在窗口期内加入
    3. 凑满 max_batch_size 或超时时触发推理
    4. 超时必发，绝不空等
    """

    def __init__(self, max_batch_size: int, max_wait_ms: float,
                 queue_capacity: int):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # 转换为秒
        self.queue = asyncio.Queue(maxsize=queue_capacity)
        self._shutdown = False

    @classmethod
    def from_config(cls, config: SchedulerConfig) -> "DynamicBatcher":
        """从配置创建调度器。"""
        return cls(
            max_batch_size=config.max_batch_size,
            max_wait_ms=config.max_wait_ms,
            queue_capacity=config.queue_capacity,
        )

    async def submit(self, request: InferenceRequest) -> asyncio.Future:
        """提交请求到队列，返回 Future 用于异步获取结果。"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request.future = future
        request.enqueue_time = time.monotonic()

        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull:
            raise QueueFullError(
                f"Request queue is full (capacity={self.queue.maxsize}). "
                f"Current size: {self.queue.qsize()}"
            )

        return future

    async def run(self, backend: Backend, preprocessor, postprocessor) -> None:
        """主循环：持续凑 batch → 推理 → 分发。"""
        while not self._shutdown:
            batch = await self._collect_batch()
            if not batch:
                continue

            # 并行预处理
            try:
                images = [req.inputs.get("image") for req in batch]
                tensors = preprocessor.process_batch(images)
            except Exception as e:
                # 预处理异常：整个 batch 失败
                for req in batch:
                    req.set_exception(e)
                continue

            # 合并成 batch 张量
            try:
                merged_input = preprocessor.merge_batch(tensors)
            except Exception as e:
                for req in batch:
                    req.set_exception(e)
                continue

            # 获取模型输入名
            input_specs = backend.get_input_specs()
            input_name = input_specs[0]["name"] if input_specs else "input"

            # Backend 推理
            infer_start = time.monotonic()
            try:
                outputs = backend.infer({input_name: merged_input})
            except Exception as e:
                for req in batch:
                    req.set_exception(e)
                continue
            infer_latency = time.monotonic() - infer_start

            # 拆分结果并分发
            batch_size = len(batch)
            output_name = list(outputs.keys())[0]
            output_tensor = outputs[output_name]

            for i, req in enumerate(batch):
                try:
                    # 取出第 i 个样本的输出
                    single_output = output_tensor[i:i + 1]  # 保持 batch 维度
                    result = postprocessor.process(single_output)
                    req.set_result(result)
                except Exception as e:
                    req.set_exception(e)

    async def _collect_batch(self) -> List[InferenceRequest]:
        """凑 batch：首请求触发窗口，后续请求在窗口期内加入，超时必发。"""
        batch = []
        deadline = None

        while len(batch) < self.max_batch_size:
            if deadline is None:
                # 等待第一个请求（阻塞）
                try:
                    req = await self.queue.get()
                except asyncio.CancelledError:
                    return []
                if self._shutdown:
                    return []
                batch.append(req)
                deadline = time.monotonic() + self.max_wait_ms
            else:
                # 非阻塞尝试获取后续请求
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

        return batch

    def shutdown(self) -> None:
        """关闭调度器。"""
        self._shutdown = True
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_scheduler.py -v
```

Expected: 4 tests passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add DynamicBatcher with collect-batch + timeout + result distribution"
```

---

## Task 7: 后处理（分类任务）

**目标:** 实现图像分类后处理（softmax + topk）。

**Files:**
- Create: `src/inference_server/core/postprocessor.py`

- [ ] **Step 1: 写测试 - 分类后处理**

Append to `tests/test_preprocessor.py` (或新建 `tests/test_postprocessor.py`):

```python
from inference_server.core.postprocessor import ClassificationPostprocessor


class TestClassificationPostprocessor:
    def test_softmax_topk(self):
        """测试分类后处理：softmax + topk"""
        postprocessor = ClassificationPostprocessor(top_k=3)

        # 模拟模型输出: [1, 5] - 5 个类别的 logits
        output = np.array([[2.0, 1.0, 0.5, 3.0, 0.1]], dtype=np.float32)
        result = postprocessor.process(output)

        assert "classes" in result
        assert "scores" in result
        assert len(result["classes"]) == 3
        assert len(result["scores"]) == 3

        # 最高分应该是索引 3
        assert result["classes"][0] == 3
        assert result["scores"][0] > result["scores"][1]
        assert abs(sum(result["scores"]) - 1.0) < 0.01  # softmax 和为 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_preprocessor.py::TestClassificationPostprocessor -v
```

Expected: ImportError

- [ ] **Step 3: 实现分类后处理器**

Create `src/inference_server/core/postprocessor.py`:

```python
"""后处理模块 - 将模型原始输出转换为结构化结果。"""

from typing import Any

import numpy as np


class Postprocessor:
    """后处理基类。"""

    def process(self, output: np.ndarray) -> dict[str, Any]:
        """处理模型输出。

        Args:
            output: 模型输出张量，shape [1, ...]（单样本）

        Returns:
            结构化结果字典
        """
        raise NotImplementedError


class ClassificationPostprocessor(Postprocessor):
    """图像分类后处理：softmax + topk。"""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def process(self, output: np.ndarray) -> dict[str, Any]:
        """分类后处理。

        Args:
            output: logits，shape [1, num_classes]

        Returns:
            {"classes": [...], "scores": [...]}
        """
        # 去除 batch 维度
        logits = output.reshape(-1)

        # softmax
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        probs = exp_logits / np.sum(exp_logits)

        # topk
        topk_indices = np.argsort(probs)[::-1][:self.top_k]
        topk_scores = probs[topk_indices]

        return {
            "classes": topk_indices.tolist(),
            "scores": topk_scores.tolist(),
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: 5 tests passed (4 preprocessor + 1 postprocessor)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add ClassificationPostprocessor with softmax and topk"
```

---

## Task 8: HTTP API 服务

**目标:** 实现 FastAPI HTTP 服务，支持健康检查、模型列表、推理接口。

**Files:**
- Create: `src/inference_server/api/schemas.py`
- Create: `src/inference_server/api/http_server.py`

- [ ] **Step 1: 实现 schemas**

Create `src/inference_server/api/schemas.py`:

```python
"""API 请求/响应模型。"""

from typing import Any, List

from pydantic import BaseModel, Field


class InferInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str = "BYTES"
    data: List[Any]


class InferOutput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: List[Any]


class InferRequest(BaseModel):
    inputs: List[InferInput]


class InferResponse(BaseModel):
    model_name: str
    outputs: List[InferOutput]


class ModelInfo(BaseModel):
    name: str
    version: str = "1"
    state: str = "READY"


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: dict[str, str]
```

- [ ] **Step 2: 实现 HTTP 服务**

Create `src/inference_server/api/http_server.py`:

```python
"""FastAPI HTTP 服务。"""

import asyncio
import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from inference_server.api.schemas import (
    ErrorResponse,
    InferRequest,
    InferResponse,
    ModelInfo,
    ModelListResponse,
)
from inference_server.config import ModelConfig
from inference_server.core.request import InferenceRequest
from inference_server.core.scheduler import DynamicBatcher


def create_http_app(
    model_manager,
    scheduler: DynamicBatcher,
) -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="Inference Server", version="0.1.0")

    @app.get("/v2/health/ready")
    async def health_ready():
        """服务就绪检查。"""
        return {"status": "ready"}

    @app.get("/v2/health/live")
    async def health_live():
        """服务存活检查。"""
        return {"status": "live"}

    @app.get("/v2/models", response_model=ModelListResponse)
    async def list_models():
        """列出所有模型。"""
        models = []
        for name, info in model_manager.list_models().items():
            models.append(ModelInfo(
                name=name,
                version="1",
                state=info["state"],
            ))
        return ModelListResponse(models=models)

    @app.post("/v2/models/{model_name}/infer", response_model=InferResponse)
    async def model_infer(model_name: str, request: InferRequest):
        """模型推理。"""
        # 获取模型配置
        model_config = model_manager.get_model_config(model_name)
        if model_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found",
            )

        # 解码图像
        try:
            input_data = request.inputs[0].data[0]
            if isinstance(input_data, str):
                # base64 编码的图像
                image_bytes = base64.b64decode(input_data)
            else:
                image_bytes = bytes(input_data)

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image data: {str(e)}",
            )

        # 创建推理请求
        req = InferenceRequest(
            model_name=model_name,
            inputs={"image": image},
        )

        # 提交到调度器
        try:
            future = await scheduler.submit(req)
            result = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference timeout",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

        # 构造响应
        return InferResponse(
            model_name=model_name,
            outputs=[
                {
                    "name": "output",
                    "shape": [1, len(result.get("classes", []))],
                    "datatype": "FP32",
                    "data": [result.get("classes", []), result.get("scores", [])],
                }
            ],
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """全局异常处理。"""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                }
            },
        )

    return app
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: add FastAPI HTTP server with health, model list, inference endpoints"
```

---

## Task 9: gRPC API 服务

**目标:** 实现 gRPC 服务。

**Files:**
- Create: `src/inference_server/api/inference.proto`
- Create: `src/inference_server/api/grpc_server.py`

- [ ] **Step 1: 定义 protobuf**

Create `src/inference_server/api/inference.proto`:

```protobuf
syntax = "proto3";

package inference;

service InferenceService {
    rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse);
    rpc ModelReady(ModelReadyRequest) returns (ModelReadyResponse);
}

message ModelInferRequest {
    string model_name = 1;
    repeated InferInput inputs = 2;
}

message InferInput {
    string name = 1;
    repeated int64 shape = 2;
    string datatype = 3;
    bytes raw_data = 4;
}

message ModelInferResponse {
    string model_name = 1;
    repeated InferOutput outputs = 2;
}

message InferOutput {
    string name = 1;
    repeated int64 shape = 2;
    string datatype = 3;
    bytes raw_data = 4;
}

message ModelReadyRequest {
    string model_name = 1;
}

message ModelReadyResponse {
    bool ready = 1;
}
```

- [ ] **Step 2: 生成 protobuf Python 代码**

```bash
cd /home/macbook/workspace/personal_project/myzone/inference-server
python -m grpc_tools.protoc \
    -I src/inference_server/api \
    --python_out=src/inference_server/api \
    --grpc_python_out=src/inference_server/api \
    src/inference_server/api/inference.proto
```

- [ ] **Step 3: 修复 protobuf 导入**

编辑生成的 `inference_pb2_grpc.py`，将 `import inference_pb2 as` 改为 `from inference_server.api import inference_pb2 as`。

- [ ] **Step 4: 实现 gRPC 服务**

Create `src/inference_server/api/grpc_server.py`:

```python
"""gRPC 服务实现。"""

import asyncio

import grpc

from inference_server.api import inference_pb2, inference_pb2_grpc
from inference_server.core.request import InferenceRequest


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC 服务实现。"""

    def __init__(self, model_manager, scheduler):
        self.model_manager = model_manager
        self.scheduler = scheduler

    async def ModelInfer(self, request, context):
        """模型推理。"""
        import cv2
        import numpy as np

        model_name = request.model_name
        model_config = self.model_manager.get_model_config(model_name)
        if model_config is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{model_name}' not found")
            return inference_pb2.ModelInferResponse()

        # 解码图像
        try:
            image_bytes = request.inputs[0].raw_data
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid image data: {str(e)}")
            return inference_pb2.ModelInferResponse()

        # 创建推理请求
        req = InferenceRequest(
            model_name=model_name,
            inputs={"image": image},
        )

        # 提交到调度器
        try:
            future = await self.scheduler.submit(req)
            result = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Inference timeout")
            return inference_pb2.ModelInferResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.ModelInferResponse()

        # 构造响应
        import json
        response_data = json.dumps(result).encode("utf-8")
        return inference_pb2.ModelInferResponse(
            model_name=model_name,
            outputs=[
                inference_pb2.InferOutput(
                    name="output",
                    shape=[1, len(result.get("classes", []))],
                    datatype="BYTES",
                    raw_data=response_data,
                )
            ],
        )

    async def ModelReady(self, request, context):
        """模型就绪检查。"""
        model_name = request.model_name
        model_config = self.model_manager.get_model_config(model_name)
        ready = model_config is not None
        return inference_pb2.ModelReadyResponse(ready=ready)


def create_grpc_server(model_manager, scheduler, port: int):
    """创建 gRPC 服务器。"""
    server = grpc.aio.server()
    servicer = InferenceServicer(model_manager, scheduler)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add gRPC server with protobuf definitions"
```

---

## Task 10: 指标系统

**目标:** 实现 Prometheus 指标暴露。

**Files:**
- Create: `src/inference_server/metrics.py`

- [ ] **Step 1: 实现指标模块**

Create `src/inference_server/metrics.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "feat: add Prometheus metrics for inference monitoring"
```

---

## Task 11: 模型管理器

**目标:** 管理模型加载、生命周期和配置。

**Files:**
- Create: `src/inference_server/model_manager.py`

- [ ] **Step 1: 实现模型管理器**

Create `src/inference_server/model_manager.py`:

```python
"""模型管理器 - 负责模型加载、生命周期和配置。"""

import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from inference_server.backend.base import Backend
from inference_server.backend.registry import get_backend
from inference_server.config import ModelConfig
from inference_server.core.postprocessor import ClassificationPostprocessor
from inference_server.core.preprocessor import ImagePreprocessor


class ModelInfo:
    """模型运行时信息。"""

    def __init__(self, config: ModelConfig, backend: Backend,
                 preprocessor: ImagePreprocessor, postprocessor):
        self.config = config
        self.backend = backend
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state = "READY"


class ModelManager:
    """模型管理器。

    负责：
    1. 从模型仓库扫描和加载模型
    2. 管理模型生命周期（LOADING → READY → UNLOADED）
    3. 提供模型配置查询
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._models: Dict[str, ModelInfo] = {}

    def load_model(self, model_name: str) -> bool:
        """加载单个模型。

        Returns:
            True if loaded successfully, False otherwise.
        """
        model_path = self.model_dir / model_name
        config_path = model_path / "config.yaml"

        if not config_path.exists():
            print(f"Model config not found: {config_path}")
            return False

        try:
            # 读取配置
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            model_config = ModelConfig(**config_data)

            # 查找模型文件
            version_dirs = sorted([d for d in model_path.iterdir() if d.is_dir()])
            if not version_dirs:
                print(f"No version directory found for model {model_name}")
                return False

            latest_version = version_dirs[-1]
            model_file = latest_version / "model.onnx"
            if not model_file.exists():
                # 尝试找其他格式的模型文件
                model_files = list(latest_version.glob("*.onnx"))
                if model_files:
                    model_file = model_files[0]
                else:
                    print(f"No model file found in {latest_version}")
                    return False

            # 创建 Backend
            backend_cls = get_backend(model_config.backend)
            backend = backend_cls()
            backend.initialize(str(model_file), model_config.backend_config)

            # 创建预处理器和后处理器
            preprocessor = ImagePreprocessor(model_config.preprocess)
            postprocessor = ClassificationPostprocessor(top_k=5)

            # 存储模型信息
            self._models[model_name] = ModelInfo(
                config=model_config,
                backend=backend,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

            print(f"Model '{model_name}' loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            return False

    def unload_model(self, model_name: str) -> None:
        """卸载模型。"""
        if model_name in self._models:
            self._models[model_name].backend.destroy()
            del self._models[model_name]
            print(f"Model '{model_name}' unloaded")

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置。"""
        info = self._models.get(model_name)
        return info.config if info else None

    def get_model_backend(self, model_name: str) -> Optional[Backend]:
        """获取模型 backend。"""
        info = self._models.get(model_name)
        return info.backend if info else None

    def get_model_preprocessor(self, model_name: str):
        """获取模型预处理器。"""
        info = self._models.get(model_name)
        return info.preprocessor if info else None

    def get_model_postprocessor(self, model_name: str):
        """获取模型后处理器。"""
        info = self._models.get(model_name)
        return info.postprocessor if info else None

    def list_models(self) -> Dict[str, dict]:
        """列出所有已加载的模型。"""
        return {
            name: {"state": info.state}
            for name, info in self._models.items()
        }

    def load_all(self, preload_list: list[str]) -> None:
        """加载指定的模型列表。"""
        for model_name in preload_list:
            self.load_model(model_name)

    def scan_and_load(self) -> None:
        """扫描模型目录并加载所有模型。"""
        if not self.model_dir.exists():
            return

        for item in self.model_dir.iterdir():
            if item.is_dir() and (item / "config.yaml").exists():
                self.load_model(item.name)
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "feat: add ModelManager for model lifecycle and config management"
```

---

## Task 12: 服务入口

**目标:** 实现 main.py，启动 HTTP + gRPC + 调度器。

**Files:**
- Create: `src/inference_server/main.py`
- Modify: `src/inference_server/api/http_server.py` (如果需要适配)

- [ ] **Step 1: 实现服务入口**

Create `src/inference_server/main.py`:

```python
"""服务入口 - 启动 HTTP + gRPC + 调度器。"""

import asyncio
import argparse
from pathlib import Path

import uvicorn
import yaml

from inference_server.api.grpc_server import create_grpc_server
from inference_server.api.http_server import create_http_app
from inference_server.config import ServiceConfig
from inference_server.core.scheduler import DynamicBatcher
from inference_server.model_manager import ModelManager


async def serve(config: ServiceConfig):
    """启动服务。"""
    # 创建模型管理器
    model_manager = ModelManager(config.models.model_dir)

    # 加载模型
    if config.models.preload:
        model_manager.load_all(config.models.preload)
    else:
        model_manager.scan_and_load()

    # 创建调度器
    scheduler = DynamicBatcher.from_config(config.scheduler)

    # 获取第一个模型的 backend/preprocessor/postprocessor 用于调度器
    # 简化：单模型场景，直接绑定
    models = model_manager.list_models()
    if not models:
        raise RuntimeError("No models loaded")

    model_name = list(models.keys())[0]
    backend = model_manager.get_model_backend(model_name)
    preprocessor = model_manager.get_model_preprocessor(model_name)
    postprocessor = model_manager.get_model_postprocessor(model_name)

    # 启动调度器（后台协程）
    scheduler_task = asyncio.create_task(
        scheduler.run(backend, preprocessor, postprocessor)
    )

    # 创建 HTTP 服务
    http_app = create_http_app(model_manager, scheduler)

    # 创建 gRPC 服务
    grpc_server = create_grpc_server(
        model_manager, scheduler, config.server.grpc_port
    )

    # 启动 gRPC
    await grpc_server.start()
    print(f"gRPC server started on port {config.server.grpc_port}")

    # 启动 HTTP (uvicorn)
    config_uvicorn = uvicorn.Config(
        app=http_app,
        host="0.0.0.0",
        port=config.server.http_port,
        log_level="info",
    )
    server = uvicorn.Server(config_uvicorn)

    print(f"HTTP server starting on port {config.server.http_port}")
    print(f"Models loaded: {list(models.keys())}")

    try:
        await server.serve()
    except asyncio.CancelledError:
        pass
    finally:
        scheduler.shutdown()
        scheduler_task.cancel()
        await grpc_server.stop(5)


def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/service.yaml",
        help="Path to service config file",
    )
    args = parser.parse_args()

    # 读取配置
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    config = ServiceConfig(**config_data)

    # 启动服务
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "feat: add main entry point for HTTP + gRPC + scheduler"
```

---

## Task 13: Docker + 配置 + 测试模型

**目标:** Dockerfile、服务配置、模型配置、测试 ONNX 模型。

**Files:**
- Create: `Dockerfile`
- Create: `configs/service.yaml`
- Create: `models/resnet50/config.yaml`
- Create: `models/resnet50/1/model.onnx` (生成简单的测试模型)

- [ ] **Step 1: 创建 Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[prod]"

# 复制代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# 暴露端口
EXPOSE 8000 8001 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v2/health/ready')" || exit 1

# 启动服务
CMD ["python", "-m", "inference_server.main", "--config", "configs/service.yaml"]
```

- [ ] **Step 2: 创建服务配置**

Create `configs/service.yaml`:

```yaml
server:
  http_port: 8000
  grpc_port: 8001
  max_concurrent_requests: 100

scheduler:
  max_batch_size: 32
  max_wait_ms: 5
  queue_capacity: 1000

metrics:
  enabled: true
  port: 8080

models:
  model_dir: "./models"
  preload:
    - "resnet50"
```

- [ ] **Step 3: 创建模型配置**

Create `models/resnet50/config.yaml`:

```yaml
name: "resnet50"
backend: "onnxruntime"

max_batch_size: 32

input:
  - name: "images"
    data_type: "FP32"
    dims: [3, 224, 224]

output:
  - name: "output"
    data_type: "FP32"
    dims: [1000]

preprocess:
  resize: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"

backend_config:
  providers:
    - name: "CPUExecutionProvider"
```

- [ ] **Step 4: 生成测试 ONNX 模型**

```python
# 运行脚本生成测试模型
import numpy as np

try:
    import onnx
    from onnx import numpy_helper, TensorProto
    from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

    # 创建一个简单的分类模型: 224x224x3 -> 1000 类
    input_info = make_tensor_value_info("images", TensorProto.FLOAT, [None, 3, 224, 224])
    output_info = make_tensor_value_info("output", TensorProto.FLOAT, [None, 1000])

    # 使用 GlobalAveragePool + MatMul 模拟分类头
    # 简化：Flatten + MatMul
    weight = numpy_helper.from_array(
        np.random.randn(3 * 224 * 224, 1000).astype(np.float32) * 0.01,
        name="weight"
    )
    bias = numpy_helper.from_array(
        np.random.randn(1000).astype(np.float32) * 0.01,
        name="bias"
    )

    node_flatten = make_node("Flatten", ["images"], ["flattened"], axis=1)
    node_gemm = make_node("Gemm", ["flattened", "weight", "bias"], ["output"])

    graph = make_graph(
        [node_flatten, node_gemm],
        "test_resnet50",
        [input_info],
        [output_info],
        initializer=[weight, bias],
    )
    model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

    with open("models/resnet50/1/model.onnx", "wb") as f:
        f.write(onnx._serialize(model))

    print("Test model generated successfully")
except ImportError:
    print("onnx package not installed, skipping test model generation")
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: add Dockerfile, configs, and test model"
```

---

## Task 14: 集成测试

**目标:** 编写端到端集成测试。

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 写集成测试**

Create `tests/test_integration.py`:

```python
import base64

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from inference_server.api.http_server import create_http_app
from inference_server.config import SchedulerConfig
from inference_server.core.scheduler import DynamicBatcher
from inference_server.model_manager import ModelManager


class TestIntegration:
    @pytest.fixture
    async def http_client(self):
        """创建测试用的 HTTP 客户端。"""
        # 需要先有测试模型被加载
        model_manager = ModelManager("models")
        model_manager.load_model("resnet50")

        scheduler = DynamicBatcher.from_config(
            SchedulerConfig(max_batch_size=4, max_wait_ms=50, queue_capacity=100)
        )

        # 启动调度器
        import asyncio
        model_name = list(model_manager.list_models().keys())[0]
        backend = model_manager.get_model_backend(model_name)
        preprocessor = model_manager.get_model_preprocessor(model_name)
        postprocessor = model_manager.get_model_postprocessor(model_name)

        task = asyncio.create_task(
            scheduler.run(backend, preprocessor, postprocessor)
        )

        app = create_http_app(model_manager, scheduler)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        scheduler.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client):
        """测试健康检查端点"""
        response = await http_client.get("/v2/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    @pytest.mark.asyncio
    async def test_list_models(self, http_client):
        """测试模型列表"""
        response = await http_client.get("/v2/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert any(m["name"] == "resnet50" for m in data["models"])

    @pytest.mark.asyncio
    async def test_infer_with_base64_image(self, http_client):
        """测试推理端点"""
        # 创建测试图像
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        response = await http_client.post(
            "/v2/models/resnet50/infer",
            json={
                "inputs": [
                    {
                        "name": "images",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [image_base64],
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "resnet50"
        assert "outputs" in data
```

- [ ] **Step 2: 运行集成测试**

```bash
pytest tests/test_integration.py -v
```

Expected: 3 tests passed

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add integration tests for HTTP API"
```

---

## Self-Review

### 1. Spec 覆盖检查

| Spec 章节 | 对应 Task | 状态 |
|-----------|----------|------|
| Backend 抽象 + 注册表 | Task 2 | ✅ |
| ONNX Backend | Task 3 | ✅ |
| 请求数据模型 | Task 4 | ✅ |
| 配置系统 | Task 4 | ✅ |
| 图像预处理 | Task 5 | ✅ |
| 动态批处理调度器 | Task 6 | ✅ |
| 后处理 | Task 7 | ✅ |
| HTTP API | Task 8 | ✅ |
| gRPC API | Task 9 | ✅ |
| 指标系统 | Task 10 | ✅ |
| 模型管理器 | Task 11 | ✅ |
| 服务入口 | Task 12 | ✅ |
| Docker 打包 | Task 13 | ✅ |
| 集成测试 | Task 14 | ✅ |

### 2. Placeholder 扫描

- ✅ 无 "TBD", "TODO", "implement later"
- ✅ 所有测试包含实际代码
- ✅ 无 "Similar to Task N" 引用

### 3. 类型一致性检查

- ✅ `DynamicBatcher.__init__` 签名在 Task 4 配置和 Task 6 实现中一致
- ✅ `Backend.infer` 签名在 base.py 和 onnx_backend.py 中一致
- ✅ `InferenceRequest` 字段在 scheduler.py 和 http_server.py 中一致

---

## 执行方式

**Plan complete and saved to `docs/superpowers/plans/2026-06-04-ascend-image-inference-service-plan.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
