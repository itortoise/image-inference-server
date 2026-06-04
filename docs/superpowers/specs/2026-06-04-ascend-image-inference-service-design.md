# 昇腾图像推理服务设计文档

> **日期**: 2026-06-04  
> **作者**: AI Assistant  
> **状态**: 已确认，待实现

---

## 1. 项目概述

### 1.1 目标

构建一个**通用的图像识别推理服务**，支持高并发请求。核心设计思想：

1. **动态批处理 (Dynamic Batching)**：收集多个请求组成 batch，让小模型在大显存机器上充分利用算力
2. **Backend 与 Serving 解耦**：模型推理后端可配置、可替换，昇腾 (Ascend NPU) 只是当前的一个 backend 实现
3. **参考开源实现**：Serving 层参考 [Triton Inference Server](https://github.com/triton-inference-server/server) 的架构设计；昇腾推理参考 [Ascend PyTorch (torch_npu)](https://gitcode.com/Ascend/pytorch) 生态下的 ONNX-CANN 路径

### 1.2 非目标

- 不支持分布式多机多卡（单机单卡即可）
- 不直接集成 NVIDIA Triton 二进制（自研 Serving 层，仅参考其设计思想）
- 不追求通用 NLP/LLM 推理（聚焦图像识别任务）

### 1.3 关键成功指标

| 指标 | 目标 |
|------|------|
| 平均 batch size | >= `max_batch_size` 的 50%（高负载时） |
| P99 延迟 | < 100ms（单请求） |
| 吞吐量 | 尽可能高（通过动态 batch 提升） |
| Backend 切换成本 | 新增 backend 只需新增一个文件，不改 serving 层 |

---

## 2. 参考项目分析

### 2.1 Triton Inference Server

[Triton Inference Server](https://github.com/triton-inference-server/server) 是 NVIDIA 开源的推理服务框架，其架构设计是我们 Serving 层的核心参考。

**核心设计思想**：

| 组件 | 职责 | 我们的借鉴方式 |
|------|------|-------------|
| **Model Repository** | 模型文件 + 配置文件的目录结构 | 完全借鉴：`models/<name>/<version>/` + `config.yaml` |
| **Dynamic Batcher** | 收集请求凑 batch，超时或满 batch 触发推理 | 借鉴算法：首请求触发窗口，后续请求在窗口期内加入，超时必发 |
| **Backend API** | C API 抽象，推理引擎通过共享库接入 | 借鉴思想：Python 抽象基类 + 装饰器注册 |
| **Model Configuration** | 配置驱动：输入输出规格、batch 策略、backend 选择 | 完全借鉴：YAML 配置 + Pydantic 校验 |
| **Metrics** | Prometheus 格式的推理指标 | 借鉴指标设计：queue/infer/output 分段延迟、batch size 分布 |
| **HTTP/gRPC 双协议** | KServe 协议 | 借鉴协议设计，但用 FastAPI/grpcio 自研 |

**不照搬的部分**：
- 不使用 Triton 的 C++ 核心和共享库机制（过重，依赖 NVIDIA 生态）
- 不支持 ensemble、BLS、decoupled backend 等高级特性（YAGNI）

### 2.2 Ascend PyTorch (torch_npu)

[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch) (`torch_npu`) 是华为昇腾的 PyTorch 适配插件。

**我们的推理路径**：

用户输入模型为 **ONNX 格式**，在昇腾 NPU 上的运行路径为：

```
ONNX 模型
    ↓
onnxruntime-cann (CANN Execution Provider)
    ↓
CANN (Compute Architecture for Neural Networks)
    ↓
昇腾 NPU (Ascend 310/910 等)
```

**关键依赖**：
- `onnxruntime-cann`：ONNX Runtime 的昇腾执行 provider（华为社区维护）
- CANN 版本兼容性：ONNX Runtime v1.20+ 对应 CANN 8.2.0

> **注意**：`torch_npu` 项目本身主要用于 PyTorch 模型训练/推理。我们的场景是 ONNX 模型推理，因此实际依赖的是 ONNX Runtime 的 CANN EP，而非 torch_npu 直接。但 `torch_npu` 项目代表了昇腾生态的软件栈，是我们的环境依赖参考。

---

## 3. 架构设计

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Requests                              │
│                              │                                       │
│            ┌─────────────────┴─────────────────┐                     │
│            ▼                                   ▼                     │
│    ┌──────────────┐                   ┌──────────────┐              │
│    │ HTTP Server  │                   │ gRPC Server  │              │
│    │  (FastAPI)   │                   │   (grpcio)   │              │
│    └──────┬───────┘                   └──────┬───────┘              │
│           │                                   │                      │
│           └───────────────┬───────────────────┘                      │
│                           ▼                                          │
│    ┌──────────────────────────────────────────────┐                 │
│    │           InferenceService                    │                 │
│    │  ┌─────────────┐    ┌─────────────────────┐  │                 │
│    │  │  Request    │───▶│   Dynamic Batcher   │  │                 │
│    │  │   Queue     │    │   (Scheduler)       │  │                 │
│    │  │(asyncio.Q)  │◀───│                     │  │                 │
│    │  └─────────────┘    └─────────────────────┘  │                 │
│    │                           │                   │                 │
│    │           ┌───────────────┼───────────────┐   │                 │
│    │           ▼               ▼               ▼   │                 │
│    │    ┌──────────┐   ┌──────────┐   ┌──────────┐│                 │
│    │    │Preprocess│   │  Backend │   │Postprocess││                 │
│    │    │(ThreadPool)   │ (ONNX/   │   │(Pluggable) │                 │
│    │    │            │   │ PyTorch) │   │           ││                 │
│    │    └──────────┘   └──────────┘   └──────────┘│                 │
│    └──────────────────────────────────────────────┘                 │
│                           │                                          │
│                           ▼                                          │
│    ┌──────────────────────────────────────────────┐                 │
│    │              Backend (抽象)                   │                 │
│    │  ┌────────────────────────────────────────┐  │                 │
│    │  │     ONNXRuntimeBackend                │  │                 │
│    │  │   ├─ CANNExecutionProvider (昇腾)     │  │                 │
│    │  │   ├─ CUDAExecutionProvider (可选)     │  │                 │
│    │  │   └─ CPUExecutionProvider (兜底)      │  │                 │
│    │  └────────────────────────────────────────┘  │                 │
│    │  ┌────────────────────────────────────────┐  │                 │
│    │  │     PyTorchBackend (预留)             │  │                 │
│    │  │   ├─ torch_npu (昇腾)                 │  │                 │
│    │  │   └─ torch.cuda (NVIDIA)              │  │                 │
│    │  └────────────────────────────────────────┘  │                 │
│    └──────────────────────────────────────────────┘                 │
│                                                                      │
│    ┌──────────────────────────────────────────────┐                 │
│    │  Metrics (Prometheus)                        │                 │
│    │  - /metrics                                  │                 │
│    └──────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 设计原则

1. **配置驱动**：模型行为完全由 `config.yaml` 决定，修改配置即可切换 backend、batch 策略、预处理参数，**无需改代码**
2. **单进程异步**：asyncio 调度 + ThreadPool 预处理，避免多进程通信开销
3. **请求-响应 1:1 绑定**：每个请求有唯一 ID + asyncio.Future，batch 拆分按索引严格对应
4. **Backend 纯增量扩展**：新增 backend 只需新建文件 + 装饰器注册，不改 serving 层

---

## 4. 目录结构

```
inference-server/
├── pyproject.toml              # 依赖管理（PEP 621）
├── Dockerfile
├── README.md
├── configs/
│   └── service.yaml            # 服务主配置
├── models/                     # 模型仓库（参考 Triton model_repository）
│   └── resnet50/
│       ├── 1/
│       │   └── model.onnx      # 模型文件
│       └── config.yaml         # 模型配置
├── src/
│   └── inference_server/       # 服务主包
│       ├── __init__.py
│       ├── main.py             # 入口：启动 HTTP + gRPC
│       ├── config.py           # Pydantic 配置模型
│       ├── api/                # 协议层
│       │   ├── http_server.py  # FastAPI HTTP 服务
│       │   ├── grpc_server.py  # gRPC 服务
│       │   ├── grpc_pb2*.py    # protobuf 生成代码
│       │   └── schemas.py      # 请求/响应 Pydantic 模型
│       ├── core/               # 核心业务逻辑
│       │   ├── scheduler.py    # 动态批处理调度器
│       │   ├── preprocessor.py # 图像预处理
│       │   ├── postprocessor.py# 后处理（可插拔）
│       │   ├── request.py      # InferenceRequest 数据类
│       │   └── request_queue.py# 请求队列封装
│       ├── backend/            # 推理后端（与 serving 完全解耦）
│       │   ├── __init__.py
│       │   ├── base.py         # Backend 抽象基类
│       │   ├── onnx_backend.py # ONNX Runtime 实现
│       │   ├── pytorch_backend.py # PyTorch 实现（预留）
│       │   └── registry.py     # Backend 注册表
│       ├── model_manager.py    # 模型加载/生命周期管理
│       └── metrics.py          # Prometheus 指标暴露
└── tests/
    ├── test_scheduler.py
    ├── test_backend.py
    ├── test_preprocessor.py
    └── test_integration.py
```

---

## 5. 配置系统

### 5.1 服务主配置 `configs/service.yaml`

```yaml
server:
  http_port: 8000
  grpc_port: 8001
  max_concurrent_requests: 100

scheduler:
  max_batch_size: 32
  max_wait_ms: 5
  queue_capacity: 1000  # 超过则拒绝请求

metrics:
  enabled: true
  port: 8080

models:
  model_dir: "./models"
  preload: ["resnet50"]  # 启动时预加载
```

### 5.2 模型配置 `models/{name}/config.yaml`

```yaml
name: "resnet50"
backend: "onnxruntime"  # 对应 backend 注册名

# 参考 Triton 的 max_batch_size + dynamic_batching
max_batch_size: 32
dynamic_batching:
  max_batch_size: 32
  max_wait_ms: 5

# 输入输出定义（参考 Triton model configuration）
input:
  - name: "images"
    data_type: "FP32"
    format: "FORMAT_NCHW"
    dims: [3, 224, 224]

output:
  - name: "output"
    data_type: "FP32"
    dims: [1000]

# 预处理配置（serving 层执行）
preprocess:
  resize: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"

# Backend 特定配置（serving 层不解析，透传给 backend）
backend_config:
  providers:
    - name: "CANNExecutionProvider"
      options:
        device_id: 0
        enable_cann_graph: true
    - name: "CPUExecutionProvider"
```

### 5.3 Pydantic 配置模型

```python
class ServerConfig(BaseModel):
    http_port: int = 8000
    grpc_port: int = 8001
    max_concurrent_requests: int = 100

class SchedulerConfig(BaseModel):
    max_batch_size: int = Field(ge=1)
    max_wait_ms: float = Field(ge=0)
    queue_capacity: int = Field(ge=1)

class ModelConfig(BaseModel):
    name: str
    backend: str
    max_batch_size: int
    dynamic_batching: DynamicBatchingConfig
    input: list[TensorConfig]
    output: list[TensorConfig]
    preprocess: PreprocessConfig
    backend_config: dict = Field(default_factory=dict)
```

**设计要点**：
- 配置分层：服务级 + 模型级
- `backend_config` 完全透传，serving 层不解析，保证 Backend 解耦
- Pydantic 启动时校验，失败即报错，不运行时出错

---

## 6. 数据流（请求生命周期）

### 6.1 时序图

```
Client          HTTP/gRPC       RequestQueue    Scheduler      Preprocessor    Backend      Postprocessor
  │                 │                │               │              │             │              │
  │──1.请求────────▶│                │               │              │             │              │
  │                 │──2.入队───────▶│               │              │             │              │
  │                 │◀─3.Future─────│               │              │             │              │
  │                 │                │               │              │             │              │
  │                 │                │               │──4.收集batch──┤             │              │
  │                 │                │               │              │             │              │
  │                 │                │               │──5.并行预处理─┤             │              │
  │                 │                │               │◀─6.tensors───│             │              │
  │                 │                │               │              │             │              │
  │                 │                │               │──7.合并[B,..]─┤             │              │
  │                 │                │               │──8.infer()───▶│             │              │
  │                 │                │               │◀─9.outputs───│             │              │
  │                 │                │               │              │             │              │
  │                 │                │               │─10.拆分结果───┤             │              │
  │                 │                │               │─11.postprocess────────────────────────────▶│
  │                 │                │               │◀─12.results────────────────────────────────│
  │                 │                │               │              │             │              │
  │                 │◀─13.Future完成─────────────────│              │             │              │
  │◀─14.响应─────────│                │               │              │             │              │
```

### 6.2 关键数据类型

```python
@dataclass
class InferenceRequest:
    request_id: str           # UUID，唯一标识
    model_name: str
    inputs: dict[str, Any]    # 原始输入（如 base64 图像）
    future: asyncio.Future    # 用于异步返回结果
    timestamp: float          # 入队时间（用于计算队列延迟）

@dataclass
class InferenceResponse:
    request_id: str
    outputs: dict[str, Any]   # 结构化结果（如 {"class": "cat", "score": 0.99}）
    error: str | None         # 错误信息
```

---

## 7. Backend 抽象层

### 7.1 抽象基类

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict


class Backend(ABC):
    """推理后端抽象基类。所有具体后端必须继承此类。"""

    @abstractmethod
    def initialize(self, model_path: str, config: Dict) -> None:
        """加载模型，准备推理环境。"""
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行推理。

        Args:
            inputs: {input_name: tensor}，张量已预处理好，shape 为 [B, ...]

        Returns:
            {output_name: tensor}，shape 为 [B, ...]
        """
        pass

    @abstractmethod
    def get_input_specs(self) -> list[dict]:
        """返回输入张量规格（name, dtype, shape）"""
        pass

    @abstractmethod
    def get_output_specs(self) -> list[dict]:
        """返回输出张量规格"""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """释放资源"""
        pass
```

### 7.2 注册机制

```python
_BACKENDS: dict[str, type[Backend]] = {}


def register_backend(name: str):
    """装饰器，注册 backend 实现。"""
    def wrapper(cls: type[Backend]) -> type[Backend]:
        _BACKENDS[name] = cls
        return cls
    return wrapper


def get_backend(name: str) -> type[Backend]:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. "
                        f"Available: {list(_BACKENDS.keys())}")
    return _BACKENDS[name]
```

### 7.3 ONNX Runtime 实现

```python
import onnxruntime as ort
from .base import Backend
from .registry import register_backend


@register_backend("onnxruntime")
class ONNXRuntimeBackend(Backend):
    def initialize(self, model_path: str, config: dict) -> None:
        providers_config = config.get("providers", [
            {"name": "CPUExecutionProvider"}
        ])
        providers = []
        for p in providers_config:
            if isinstance(p, dict):
                providers.append((p["name"], p.get("options", {})))
            else:
                providers.append(p)

        self.session = ort.InferenceSession(model_path, providers=providers)

        # 从 ONNX 模型自动获取输入输出规格
        self._input_specs = [
            {
                "name": inp.name,
                "dtype": self._onnx_type_to_numpy(inp.type),
                "shape": list(inp.shape) if inp.shape else [-1],
            }
            for inp in self.session.get_inputs()
        ]
        self._output_specs = [
            {
                "name": out.name,
                "dtype": self._onnx_type_to_numpy(out.type),
                "shape": list(out.shape) if out.shape else [-1],
            }
            for out in self.session.get_outputs()
        ]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, inputs)
        output_names = [o.name for o in self.session.get_outputs()]
        return dict(zip(output_names, outputs))

    def get_input_specs(self) -> list[dict]:
        return self._input_specs

    def get_output_specs(self) -> list[dict]:
        return self._output_specs

    def destroy(self) -> None:
        del self.session
```

### 7.4 模型加载流程

```
config.yaml 中 backend: "onnxruntime"
            ↓
ModelManager 读取配置
            ↓
registry.get_backend("onnxruntime") → ONNXRuntimeBackend 类
            ↓
实例化 → initialize(model_path, backend_config)
            ↓
状态变为 READY，等待 scheduler 调用 infer()
```

---

## 8. 动态批处理调度器

### 8.1 核心算法

```python
class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: float,
                 queue_capacity: int):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=queue_capacity)
        self._shutdown = False

    async def submit(self, request: InferenceRequest) -> asyncio.Future:
        """API 层调用：提交请求，返回 Future。"""
        future = asyncio.get_event_loop().create_future()
        request.future = future
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull:
            raise QueueFullError("Request queue is full")
        return future

    async def run(self, backend: Backend, preprocessor, postprocessor):
        """主循环：持续凑 batch → 推理 → 分发。"""
        while not self._shutdown:
            batch = await self._collect_batch()
            if not batch:
                continue

            # 1. 并行预处理
            tensors = await self._preprocess_batch(batch, preprocessor)

            # 2. 过滤预处理失败的请求
            valid_requests = []
            valid_tensors = []
            for req, tensor in zip(batch, tensors):
                if isinstance(tensor, Exception):
                    req.future.set_exception(tensor)
                else:
                    valid_requests.append(req)
                    valid_tensors.append(tensor)

            if not valid_requests:
                continue

            # 3. 合并成 [B, C, H, W]
            merged_input = self._merge_tensors(valid_tensors)

            # 4. Backend 推理
            start = time.monotonic()
            try:
                outputs = backend.infer(merged_input)
            except Exception as e:
                # 整个 batch 失败
                for req in valid_requests:
                    req.future.set_exception(e)
                metrics.inference_failure.inc(len(valid_requests))
                continue
            infer_latency = time.monotonic() - start

            # 5. 按 batch 索引拆分
            split_outputs = self._split_outputs(outputs, len(valid_requests))

            # 6. 后处理 + 设置 Future 结果
            for req, out in zip(valid_requests, split_outputs):
                try:
                    result = postprocessor.process(out)
                    req.future.set_result(result)
                except Exception as e:
                    req.future.set_exception(e)

            # 7. 记录指标
            metrics.batch_size.observe(len(valid_requests))
            metrics.inference_latency.observe(infer_latency)

    async def _collect_batch(self) -> list[InferenceRequest]:
        """凑 batch：首请求触发窗口，后续请求在窗口期内加入，超时必发。"""
        batch = []
        deadline = None

        while len(batch) < self.max_batch_size:
            if deadline is None:
                # 等待第一个请求（阻塞）
                req = await self.queue.get()
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
```

### 8.2 凑 batch 时序示例

```
时间轴 →

[t0] Req1 到达 ──────────────────────────────────────┐
     队列空，阻塞等待 Req1                              │
     收到 Req1，batch=[Req1]，设 deadline=t0+5ms       │
                                                       │
[t1=t0+2ms] Req2 到达 ────────────────────────────────┤
     在 5ms 窗口内，batch=[Req1, Req2]                 │
                                                       │
[t2=t0+3ms] Req3 到达 ────────────────────────────────┤
     在 5ms 窗口内，batch=[Req1, Req2, Req3]           │
                                                       │
[t3=t0+5ms] 超时 ─────────────────────────────────────┤
     batch_size=3 < max_batch_size=32，但超时已到       │
     发 batch → 预处理 → 推理 → 拆分 → 返回            │
                                                       │
[t4] Req4 到达 ───────────────────────────────────────┘
     开始新的 batch 窗口
```

### 8.3 关键设计点

1. **首请求触发**：没有请求时调度器阻塞等待，避免空转
2. **超时必发**：绝不空等超过 `max_wait_ms`，避免延迟爆炸
3. **部分失败处理**：单个请求预处理失败不影响同 batch 其他请求
4. **队列满拒绝**：超过 `queue_capacity` 返回 429，client 可重试

---

## 9. 错误处理

### 9.1 分段错误处理策略

| 阶段 | 错误类型 | 处理方式 | HTTP 状态码 |
|------|---------|---------|------------|
| 请求接收 | JSON 格式错误、图像解码失败、尺寸超限 | 立即返回，不入队 | 400 |
| 入队 | 队列已满 | 拒绝请求 | 429 |
| 预处理 | 单张图像处理异常 | 仅该请求失败，同 batch 其他继续 | 500 (该请求) |
| Backend 推理 | 模型内部错误、NPU 异常、OOM | 整个 batch 失败 | 500 |
| 后处理 | 输出维度不符 | 单个请求失败 | 500 (该请求) |
| 模型加载 | 文件不存在、backend 初始化失败 | 模型状态 UNAVAILABLE | 503 |

### 9.2 模型状态机

```
         ┌───────────┐
         │  UNLOADED │
         └─────┬─────┘
               │ load()
               ▼
         ┌───────────┐
    ┌───▶│  LOADING  │
    │    └─────┬─────┘
    │          │ success
    │          ▼
    │    ┌───────────┐     unload()
    └────┤   READY   │◀────────┐
         └─────┬─────┘         │
               │ failure       │
               ▼               │
         ┌───────────┐         │
         │  FAILED   │─────────┘
         └───────────┘   retry
```

### 9.3 错误响应格式（统一）

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Image decode failed: unsupported format",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

---

## 10. 监控指标

### 10.1 暴露的 Prometheus 指标

参考 Triton 指标设计：

```
# 请求计数（按模型、状态）
inference_request_total{model="resnet50",status="success"} 1024
inference_request_total{model="resnet50",status="failure"}  3

# 推理次数（一个 batch 为 N 算 N 次）
inference_count_total{model="resnet50"}  8192

# batch 执行次数
inference_exec_count_total{model="resnet50"} 512

# 平均 batch size = inference_count / inference_exec_count = 16

# 队列深度（Gauge，实时）
pending_request_count{model="resnet50"} 7

# 分段延迟（Histogram，单位秒）
queue_duration_seconds_bucket{model="resnet50",le="0.005"} 980
queue_duration_seconds_bucket{model="resnet50",le="0.01"}  995
queue_duration_seconds_bucket{model="resnet50",le="+Inf"}  1000

compute_input_duration_seconds_bucket{model="resnet50",le="0.001"} ...
compute_infer_duration_seconds_bucket{model="resnet50",le="0.01"} ...
compute_output_duration_seconds_bucket{model="resnet50",le="0.001"} ...

# batch 大小分布
batch_size_histogram_bucket{model="resnet50",le="4"}  50
batch_size_histogram_bucket{model="resnet50",le="8"}  120
batch_size_histogram_bucket{model="resnet50",le="16"} 350
batch_size_histogram_bucket{model="resnet50",le="32"} 512
```

### 10.2 调优核心公式

```
avg_batch_size = inference_count / inference_exec_count
```

**调优策略**：
- `avg_batch_size` << `max_batch_size` 的一半：请求量不足或超时太短，尝试增加 `max_wait_ms`
- `avg_batch_size` ≈ `max_batch_size`：可以增加 `max_batch_size` 进一步压榨显存
- `pending_request_count` 持续高：请求积压，考虑扩容或降低延迟

---

## 11. API 接口定义

### 11.1 HTTP API（FastAPI）

#### 健康检查

```
GET /v2/health/ready
GET /v2/health/live
```

#### 模型列表

```
GET /v2/models
Response:
{
  "models": [
    {"name": "resnet50", "version": "1", "state": "READY"}
  ]
}
```

#### 模型推理

```
POST /v2/models/{model_name}/infer
Content-Type: application/json

Request:
{
  "inputs": [
    {
      "name": "images",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["/9j/4AAQ...base64encodedimage..."]
    }
  ]
}

Response:
{
  "model_name": "resnet50",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 1000],
      "datatype": "FP32",
      "data": [[0.01, 0.85, 0.02, ...]]
    }
  ]
}
```

### 11.2 gRPC API

```protobuf
syntax = "proto3";

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
```

---

## 12. Docker 打包

### 12.1 Dockerfile

```dockerfile
# 基于昇腾基础镜像
FROM ascend-pytorch:8.0.RC2

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层
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
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# 启动服务
CMD ["python", "-m", "inference_server.main", "--config", "configs/service.yaml"]
```

### 12.2 pyproject.toml

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

---

## 13. 实现优先级

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | `backend/base.py` + `onnx_backend.py` | Backend 抽象 + ONNX 昇腾实现 |
| P0 | `core/scheduler.py` | 动态批处理调度器（核心） |
| P0 | `api/http_server.py` | FastAPI HTTP 服务 |
| P0 | `core/preprocessor.py` | 图像预处理 |
| P1 | `api/grpc_server.py` | gRPC 服务 |
| P1 | `metrics.py` | Prometheus 指标 |
| P1 | `model_manager.py` | 模型生命周期管理 |
| P1 | `core/postprocessor.py` | 后处理（分类/检测） |
| P2 | `tests/` | 单元测试 + 集成测试 |
| P2 | Dockerfile | 容器化打包 |

---

## 14. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| CANN EP 对某算子不支持 | 模型无法运行 | fallback 到 CPUExecutionProvider |
| 昇腾驱动/固件版本不匹配 | 服务无法启动 | Dockerfile 固定基础镜像版本 |
| 动态 batch 延迟不可控 | P99 延迟升高 | 可配置 `max_wait_ms`，上线后调优 |
| 图像预处理成为瓶颈 | 吞吐量下降 | ThreadPoolExecutor 并行 + 监控 |
| 显存 OOM | 服务崩溃 | 设置 `max_batch_size` 上限 + 异常捕获 |

---

## 15. 附录

### 15.1 相关链接

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Triton Dynamic Batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher)
- [Triton Metrics](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md)
- [Ascend PyTorch (torch_npu)](https://gitcode.com/Ascend/pytorch)
- [ONNX Runtime CANN EP](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)

### 15.2 术语表

| 术语 | 说明 |
|------|------|
| CANN | Compute Architecture for Neural Networks，华为昇腾计算架构 |
| EP | Execution Provider，ONNX Runtime 的执行后端抽象 |
| Dynamic Batching | 动态批处理：运行时收集请求组成 batch |
| Backend | 推理后端：负责张量进张量出的模型执行 |
| Serving | 服务层：负责 HTTP/gRPC 协议、调度、预处理 |
