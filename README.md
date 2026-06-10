# Inference Server — 通用图像识别推理服务

一个支持**动态批处理（Dynamic Batching）**的通用图像识别推理服务，采用**配置驱动**架构，Backend 与 Serving 完全解耦，支持 HTTP/gRPC 双协议暴露。

> **设计参考**：[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) 的模型仓库与配置分层思想。

---

## 目录

1. [核心特性](#核心特性)
2. [架构概览](#架构概览)
3. [环境要求](#环境要求)
4. [安装构建](#安装构建)
5. [快速开始](#快速开始)
6. [部署方式](#部署方式)
7. [配置详解](#配置详解)
8. [API 使用指南](#api-使用指南)
9. [测试](#测试)
10. [扩展开发](#扩展开发)
11. [监控指标](#监控指标)
12. [已知限制](#已知限制)
13. [项目结构](#项目结构)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **动态批处理** | 收集多个请求自动组 batch，超时必发，提升 GPU/CPU 利用率 |
| **配置驱动** | 模型行为由 `config.yaml` 决定，改配置即可切换后端、batch 策略、预处理 |
| **Backend 解耦** | 新增推理引擎只需继承 `Backend` 基类并注册，与协议层完全隔离 |
| **双协议** | 同时提供 RESTful HTTP 和高效 gRPC 接口 |
| **ONNX 自动适配** | 自动检测模型是否支持动态 batch，支持则合并推理，不支持则逐张兼容 |
| **硬件弹性** | 通过 ONNX Runtime Provider 配置支持 CPU / NVIDIA GPU / TensorRT |

---

## 架构概览

```
┌─────────────┐     ┌─────────────┐
│ HTTP Client │     │ gRPC Client │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────┐
│      FastAPI HTTP Server            │
│      gRPC AIO Server                │
│            │                        │
│     ┌──────┴──────┐                │
│     ▼             ▼                │
│  ┌────────┐   ┌──────────┐        │
│  │ submit │   │ submit   │        │
│  │(Future)│   │(Future)  │        │
│  └───┬────┘   └────┬─────┘        │
└──────┼─────────────┼──────────────┘
       │             │
       ▼             ▼
┌─────────────────────────────────────┐
│     DynamicBatcher 调度器           │
│  ┌─────────────────────────────┐   │
│  │  凑 batch：首请求触发窗口   │   │
│  │  max_batch_size / max_wait  │   │
│  └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │ batch
                  ▼
┌─────────────────────────────────────┐
│   ImagePreprocessor 图像预处理       │
│   (resize → normalize → HWC→CHW)   │
└─────────────────┬───────────────────┘
                  │ tensor list
                  ▼
┌─────────────────────────────────────┐
│      ONNXRuntimeBackend             │
│   ┌─────────────────────────────┐   │
│   │  动态 batch: concat→run→split│   │
│   │  静态 batch: 逐个 infer_single │   │
│   └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │ raw outputs
                  ▼
┌─────────────────────────────────────┐
│  ClassificationPostprocessor        │
│      (softmax + top-k)              │
└─────────────────┬───────────────────┘
                  │ result dict
                  ▼
            ┌──────────┐
            │  Future  │
            │ .set_result
            └────┬─────┘
                 │ 异步返回
                 ▼
           HTTP/gRPC Response
```

---

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | 必需 |
| CUDA | 12.x + cuDNN 9.x | **GPU 推理可选**，CPU 可直接运行 |
| Docker | 任意 | 容器化部署可选 |
| uv | 最新 | 推荐包管理器（Rust 编写，极速） |

---

## 安装构建

### 前置条件

部分 Python 包依赖系统库和编译工具，安装前请确保系统环境满足要求：

**Ubuntu/Debian：**

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 build-essential
```

**macOS：**

```bash
# 安装 Xcode 命令行工具（包含 gcc）
xcode-select --install
```

**缺少系统库的典型报错：**

```
ImportError: libGL.so.1: cannot open shared object file    # opencv 需要
ImportError: libgthread-2.0.so.0: cannot open shared object  # opencv 需要
error: Microsoft Visual C++ 14.0 is required               # grpcio 编译需要
```

### 方式一：pip

#### 1. 配置国内镜像（国内环境必需）

```bash
# 创建 pip 配置文件
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
trusted-host = mirrors.tuna.tsinghua.edu.cn
EOF
```

其他国内镜像源：
- 阿里云：`https://mirrors.aliyun.com/pypi/simple/`
- 腾讯云：`https://mirrors.cloud.tencent.com/pypi/simple/`
- 豆瓣：`https://pypi.doubanio.com/simple/`

#### 2. 安装依赖

```bash
cd /path/to/inference-server

# 创建虚拟环境（推荐，避免污染系统 Python）
python -m venv .venv
source .venv/bin/activate

# 安装项目 + 开发依赖
pip install -e ".[dev]"
```

#### 3. 启动服务

```bash
source .venv/bin/activate
python -m inference_server.main --config configs/service.yaml
```

### 方式二：uv（推荐，项目已锁定依赖版本）

> ⚠️ **重要**：uv 默认从 PyPI (`pypi.org`) 下载包，**不会读取** pip 的镜像配置。国内环境必须先配置 uv 的镜像源。

#### 1. 安装 uv

```bash
# 方式 A：官方脚本（从 GitHub 下载）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方式 B：pip 安装（从 PyPI，不经过 GitHub）
pip install uv
```

#### 2. 配置国内镜像

```bash
# 设置环境变量（当前终端生效）
export UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 或写入 shell 配置文件永久生效
echo 'export UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple' >> ~/.bashrc
```

#### 3. 安装依赖

```bash
cd /path/to/inference-server

# 同步依赖（--dev 包含测试依赖，--frozen 使用 uv.lock 锁定版本）
UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  uv sync --dev --frozen

# 或直接用 uv 的 pip 兼容模式安装（会自动走 UV_DEFAULT_INDEX）
uv venv --python python3
uv pip install -e ".[dev]"
```

#### 4. 启动服务

```bash
# 使用 uv run（自动激活虚拟环境）
uv run python -m inference_server.main --config configs/service.yaml

# 或使用 CLI 入口
uv run inference-server --config configs/service.yaml
```

### pip vs uv 的选择指南

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 有 pip 镜像配置，无 uv | `pip install -e ".[dev]"` | 直接用现有配置 |
| 需要可复现的锁定依赖 | `uv sync --frozen` | `uv.lock` 精确锁定所有版本 |
| 国内网络环境 | 两者都需要配置镜像 | uv 用 `UV_DEFAULT_INDEX`，pip 用 `pip.conf` |
| Docker 构建 | `uv sync --frozen` | 更快，支持分层缓存 |

### GPU 支持

#### 切换到 GPU 版本

```bash
# pip 方式
pip uninstall onnxruntime -y
pip install onnxruntime-gpu

# uv 方式
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu
```

#### 验证 GPU 可用

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 期望输出包含 'CUDAExecutionProvider'
```

#### Provider 对照

| Provider | 说明 | 依赖 |
|----------|------|------|
| `CPUExecutionProvider` | CPU 推理，零额外依赖 | `onnxruntime` |
| `CUDAExecutionProvider` | NVIDIA GPU 加速 | `onnxruntime-gpu` + CUDA 12.x + cuDNN 9.x |
| `TensorrtExecutionProvider` | TensorRT 极致加速 | `onnxruntime-gpu` + TensorRT |

---

## 快速开始

### 1. 准备模型（项目已预置）

```bash
# 查看已有模型
ls models/
# └── resnet34/          ImageNet 图像分类
# └── mnist_pretrained/  MNIST 手写数字识别
# └── resnet50/          另一个图像分类模型
```

### 2. 启动服务

**如果你用 pip 安装（在虚拟环境中）：**

```bash
source .venv/bin/activate
python -m inference_server.main --config configs/service.yaml

# 或使用 CLI 入口
inference-server --config configs/service.yaml
```

**如果你用 uv 安装：**

```bash
# 在项目目录下执行（uv 会自动使用 .venv）
uv run python -m inference_server.main --config configs/service.yaml

# 或使用 CLI 入口
uv run inference-server --config configs/service.yaml
```

服务启动后：

| 端点 | 地址 | 说明 |
|------|------|------|
| HTTP API | http://localhost:8000 | RESTful 推理接口 |
| gRPC API | localhost:8001 | 高性能推理接口 |
| 健康检查 | http://localhost:8000/v2/health/ready | 就绪探针 |
| 存活检查 | http://localhost:8000/v2/health/live | 存活探针 |

### 3. 验证服务

```bash
# 查看已加载的模型
curl http://localhost:8000/v2/models

# 健康检查
curl http://localhost:8000/v2/health/ready
```

### 4. 发送推理请求

**ResNet34 图像分类**（ImageNet 1000 类）：

```bash
curl -X POST http://localhost:8000/v2/models/resnet34/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["<base64编码的RGB图像>"]
    }]
  }'
```

**MNIST 手写数字识别**：

```bash
curl -X POST http://localhost:8000/v2/models/mnist_pretrained/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "Input3",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["<base64编码的灰度图像>"]
    }]
  }'
```

Python 客户端示例见 [API 使用指南](#api-使用指南) 章节。

---

## 部署方式

### 本地部署

```bash
# 前台运行（开发调试）
python -m inference_server.main --config configs/service.yaml

# 后台运行（生产）
nohup python -m inference_server.main --config configs/service.yaml > server.log 2>&1 &
```

### Docker 部署（CPU）

```bash
# 构建镜像
docker build -t inference-server:latest .

# 运行（CPU 模式）
docker run -d \
  -p 8000:8000 \
  -p 8001:8001 \
  --name inference-server \
  inference-server:latest
```

### Docker 部署（GPU）

```bash
# 确保宿主机已安装 NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 构建时可选：基于 GPU 基础镜像（需修改 Dockerfile 安装 onnxruntime-gpu）
# 或运行时挂载 CUDA 驱动

docker run -d \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/configs:/app/configs:ro \
  --name inference-server \
  inference-server:latest
```

### Docker 挂载外部模型目录

生产环境通常将模型目录挂载为卷，而非打包进镜像：

```bash
docker run -d \
  -p 8000:8000 \
  -p 8001:8001 \
  -v /path/to/your/models:/app/models:ro \
  -v /path/to/your/configs/service.yaml:/app/configs/service.yaml:ro \
  inference-server:latest
```

> 建议将 `models/` 和 `configs/` 作为 Docker Volume 挂载，实现**镜像与模型分离**，模型更新无需重新构建镜像。

---

## 配置详解

### 配置分层

| 配置级别 | 文件 | 控制内容 |
|---------|------|---------|
| **服务级** | `configs/service.yaml` | 端口、调度器参数、预加载模型列表、指标开关 |
| **模型级** | `models/{name}/config.yaml` | backend 类型、输入输出 shape、预处理参数、GPU provider |

### 服务配置示例（`configs/service.yaml`）

```yaml
server:
  http_port: 8000          # HTTP 服务端口
  grpc_port: 8001          # gRPC 服务端口
  max_concurrent_requests: 100

scheduler:
  max_batch_size: 16       # 最大 batch 大小
  max_wait_ms: 5           # 凑 batch 最大等待时间（毫秒）
  queue_capacity: 1000     # 请求队列容量

metrics:
  enabled: true            # Prometheus 指标开关
  port: 8080               # 指标暴露端口

models:
  model_dir: "./models"    # 模型仓库根目录
  preload:                 # 启动时预加载的模型列表
    - "resnet34"
    - "mnist_pretrained"
```

### 模型配置示例（`models/resnet34/config.yaml`）

```yaml
name: "resnet34"
backend: "onnxruntime"
max_batch_size: 16

input:
  - name: "input"           # ONNX 模型输入节点名
    data_type: "FP32"
    dims: [3, 224, 224]    # 单张输入 shape（不含 batch 维度）

output:
  - name: "output"          # ONNX 模型输出节点名
    data_type: "FP32"
    dims: [1000]

preprocess:
  resize: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"       # 可选: RGB / GRAY / BGR

backend_config:
  providers:
    - name: "CUDAExecutionProvider"   # NVIDIA GPU
      options:
        device_id: 0
    - name: "CPUExecutionProvider"    # 兜底 fallback
```

### 查询 ONNX 模型的输入输出节点

```python
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx")
print("Inputs:",  [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])
print("Input shapes:", [i.shape for i in sess.get_inputs()])
```

---

## API 使用指南

### HTTP API

#### 健康检查

```bash
# 就绪检查（模型已加载）
curl http://localhost:8000/v2/health/ready
# → {"status": "ready"}

# 存活检查（进程存活）
curl http://localhost:8000/v2/health/live
# → {"status": "live"}
```

#### 列出模型

```bash
curl http://localhost:8000/v2/models
```

响应：
```json
{
  "models": [
    {"name": "resnet34", "version": "1", "state": "READY"},
    {"name": "mnist_pretrained", "version": "1", "state": "READY"}
  ]
}
```

#### 推理请求

**POST** `/v2/models/{model_name}/infer`

请求体：
```json
{
  "inputs": [
    {
      "name": "input",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["<base64编码的图像>"]
    }
  ]
}
```

响应体：
```json
{
  "model_name": "resnet34",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 1000],
      "datatype": "FP32",
      "data": [
        {"classes": [973, 599, 539, 109, 107], "scores": [0.85, 0.12, 0.02, 0.005, 0.003]}
      ]
    }
  ]
}
```

#### 完整 Python 客户端

```python
import base64
import cv2
import requests


def infer_image(model_name: str, image_path: str, server_url: str = "http://localhost:8000"):
    """发送图像推理请求。"""
    # 读取图像（OpenCV 默认 BGR）
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB

    # 编码为 PNG base64
    _, buffer = cv2.imencode(".png", image)
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    # 发送请求
    resp = requests.post(
        f"{server_url}/v2/models/{model_name}/infer",
        json={
            "inputs": [{
                "name": "input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_b64]
            }]
        }
    )
    resp.raise_for_status()
    return resp.json()


# 使用
result = infer_image("resnet34", "cat.jpg")
top_class = result["outputs"][0]["data"][0]["classes"][0]
top_score = result["outputs"][0]["data"][0]["scores"][0]
print(f"Top class: {top_class} (score: {top_score:.4f})")
```

### gRPC API

```python
import grpc
from inference_server.api import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("localhost:8001")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

# 推理
request = inference_pb2.ModelInferRequest(
    model_name="resnet34",
    inputs=[inference_pb2.InferInput(
        name="input",
        shape=[1],
        datatype="BYTES",
        raw_data=image_bytes  # PNG 原始字节
    )]
)
response = stub.ModelInfer(request)
print(response)
```

### 重新生成 gRPC 代码（修改 .proto 后）

```bash
# 从 proto 文件生成 Python 代码
python -m grpc_tools.protoc \
  -I src/inference_server/api \
  --python_out=src/inference_server/api \
  --grpc_python_out=src/inference_server/api \
  src/inference_server/api/inference.proto
```

---

## 测试

### 测试分类

| 测试类型 | 命令 | 前提条件 | 说明 |
|---------|------|---------|------|
| **单元测试** | `pytest tests/ -v` | 安装 `[dev]` 依赖 | 不依赖外部服务，覆盖核心组件 |
| **端到端测试** | `python scripts/test_mnist_e2e.py` | 无需服务 | 本地完整链路测试 |
| **HTTP 集成测试** | `python scripts/test_mnist_e2e.py --mode http` | 服务已启动 | 测试 HTTP API |
| **ResNet34 测试** | `python scripts/test_resnet34.py` | 无需服务 | 测试 ResNet34 模型推理 |
| **静态 Batch 测试** | `python scripts/test_static_batch_model.py` | 无需服务 | 验证静态 batch 模型兼容性 |
| **性能压测** | `python scripts/benchmark.py` | 服务已启动 | 测量吞吐量和延迟 |

### 运行单元测试

```bash
# 全部单元测试
pytest tests/ -v

# 带覆盖率报告
pytest tests/ -v --cov=src/inference_server --cov-report=term-missing

# 仅测试调度器
pytest tests/test_scheduler.py -v

# 仅测试 Backend
pytest tests/test_backend.py -v
```

### 运行端到端测试

```bash
# 完整测试（单张 + 批量 + 调度器）
python scripts/test_mnist_e2e.py --mode all

# 仅测试单张图片完整流程
python scripts/test_mnist_e2e.py --mode single

# 仅测试调度器批量推理
python scripts/test_mnist_e2e.py --mode batch

# 测试 HTTP 端点（需先启动服务）
python scripts/test_mnist_e2e.py --mode http
```

### 性能压测

```bash
# 先启动服务
python -m inference_server.main --config configs/service.yaml

# 运行压测（默认 100 请求，并发 10）
python scripts/benchmark.py --model mnist_pretrained --concurrency 10 --total 100

# 更高并发压测
python scripts/benchmark.py --model mnist_pretrained --concurrency 50 --total 1000
```

压测输出示例：
```
📊 压测结果
  总请求数:     1000
  并发数:       50
  成功:         1000
  失败:         0
  总耗时:       2.341s
  吞吐量:       427.15 req/s
  平均延迟:     115.23ms
  P50 延迟:     108.45ms
  P99 延迟:     245.67ms
```

---

## 扩展开发

### 1. 添加自己的 ONNX 模型

#### 步骤 1：创建模型目录

```bash
mkdir -p models/my_model/1
cp your_model.onnx models/my_model/1/model.onnx
```

#### 步骤 2：编写模型配置

创建 `models/my_model/config.yaml`：

```yaml
name: "my_model"
backend: "onnxruntime"
max_batch_size: 16

input:
  - name: "input"           # ONNX 输入节点名
    data_type: "FP32"
    dims: [3, 224, 224]

output:
  - name: "output"          # ONNX 输出节点名
    data_type: "FP32"
    dims: [1000]

preprocess:
  resize: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"

backend_config:
  providers:
    - name: "CUDAExecutionProvider"
      options:
        device_id: 0
    - name: "CPUExecutionProvider"
```

#### 步骤 3：注册到服务配置

编辑 `configs/service.yaml`，在 `preload` 列表中加入 `my_model`：

```yaml
models:
  preload:
    - "resnet34"
    - "mnist_pretrained"
    - "my_model"
```

#### 步骤 4：重启服务

```bash
python -m inference_server.main --config configs/service.yaml
```

### 2. 导出 PyTorch 模型为 ONNX

项目提供 ResNet34 导出脚本作为参考：

```bash
# 导出 ResNet34（ImageNet 预训练权重）
python scripts/export_resnet34.py

# 导出自定义 PyTorch 模型
python scripts/export_mnist_onnx.py  # 纯 numpy + onnx 创建示例
```

导出要点：

```python
import torch

# 关键：设置 dynamic_axes 使 batch 维度可变，支持动态批处理
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=11,
)
```

### 3. 添加新的推理 Backend

所有 Backend 通过装饰器注册，与协议层完全隔离。

```python
# src/inference_server/backend/my_backend.py
import numpy as np
from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend


@register_backend("mybackend")
class MyBackend(Backend):
    """自定义推理后端。"""

    def initialize(self, model_path: str, config: dict) -> None:
        """加载模型。"""
        self.model = load_your_model(model_path)
        self.device = config.get("device", "cpu")

    def infer_single(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """单张推理。inputs shape 为 [1, ...]。"""
        # 实现推理逻辑
        return {"output": self.model(inputs["input"])}

    def infer_batch(self, inputs_list: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        """批量推理。可选覆盖，默认逐个调用 infer_single。"""
        # 如果支持合并推理，在这里实现
        merged = np.concatenate([inp["input"] for inp in inputs_list], axis=0)
        outputs = self.model(merged)
        # 拆分结果
        return [{"output": outputs[i:i+1]} for i in range(len(inputs_list))]

    def get_input_specs(self) -> list[dict]:
        return [{"name": "input", "dtype": "FP32", "shape": [3, 224, 224]}]

    def get_output_specs(self) -> list[dict]:
        return [{"name": "output", "dtype": "FP32", "shape": [1000]}]

    def destroy(self) -> None:
        """释放资源。"""
        del self.model
```

然后在模型配置中使用：

```yaml
backend: "mybackend"
backend_config:
  device: "cuda:0"
```

### 4. 添加新的后处理器

当前默认使用 `ClassificationPostprocessor`（softmax + top-k）。如需支持检测、分割等任务：

```python
# src/inference_server/core/postprocessor.py

class DetectionPostprocessor(Postprocessor):
    """目标检测后处理：解析边界框 + NMS。"""

    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def process(self, output: np.ndarray) -> dict[str, Any]:
        """处理检测模型输出。

        Args:
            output: 模型输出，形状取决于模型（如 [1, 84, 8400] YOLOv8 格式）

        Returns:
            {"boxes": [...], "scores": [...], "classes": [...]}
        """
        # 实现解析逻辑
        return {"boxes": [], "scores": [], "classes": []}
```

> **注意**：当前版本 `ModelManager` 硬编码使用 `ClassificationPostprocessor`。多任务支持需要修改 `ModelManager` 根据模型配置选择对应后处理器。

### 5. 自定义预处理

预处理通过 `PreprocessConfig` 配置驱动，支持以下参数：

```yaml
preprocess:
  resize: [224, 224]           # 目标尺寸 [H, W]
  mean: [0.485, 0.456, 0.406]  # 归一化均值
  std: [0.229, 0.224, 0.225]   # 归一化标准差
  pixel_format: "RGB"          # RGB / GRAY / BGR
```

如需更复杂的预处理（如 Letterbox、随机裁剪等），可继承 `ImagePreprocessor`：

```python
from inference_server.core.preprocessor import ImagePreprocessor

class LetterboxPreprocessor(ImagePreprocessor):
    """保持宽高比的 Letterbox 预处理（YOLO 风格）。"""

    def process(self, image: np.ndarray) -> np.ndarray:
        # 实现 letterbox resize + padding
        return super().process(image)  # 或完全自定义
```

---

## 监控指标

服务内置 Prometheus 指标收集（代码位于 `src/inference_server/metrics.py`）：

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `inference_request_total` | Counter | 推理请求总数（按状态分类） |
| `inference_count_total` | Counter | 实际推理次数 |
| `inference_exec_count_total` | Counter | batch 执行次数 |
| `pending_request_count` | Gauge | 队列中待处理请求数 |
| `queue_duration_seconds` | Histogram | 请求在队列中的等待时间 |
| `compute_infer_duration_seconds` | Histogram | 实际推理计算时间 |
| `batch_size_histogram` | Histogram | batch 大小分布 |

> **当前状态**：指标收集代码已就绪，但尚未在 HTTP Server 中集成暴露。如需启用，需在 `http_server.py` 中启动 Prometheus HTTP server 并在调度器中埋点。

---

## 已知限制

1. **单模型调度器**：当前 `DynamicBatcher` 在启动时只绑定第一个模型的 `backend/preprocessor/postprocessor`。同时加载多个不同预处理/后处理的模型时，所有请求会走同一个处理流水线。** workaround**：为不同模型启动独立的服务实例。

2. **后处理器硬编码**：`ModelManager` 目前固定使用 `ClassificationPostprocessor(top_k=5)`，不支持自动根据任务类型选择后处理器。

3. **Prometheus 指标未集成**：指标定义已完成，但未在请求链路中实际埋点和暴露。

4. **输入格式单一**：HTTP API 目前只支持 base64 编码的图像输入，不支持原始二进制、multipart/form-data 或 URL。

---

## 项目结构

```
inference-server/
├── pyproject.toml              # 依赖管理（pip/uv）
├── uv.lock                     # uv 锁定依赖版本
├── Dockerfile                  # Docker 构建
├── configs/
│   └── service.yaml            # 服务主配置
├── models/                     # 模型仓库（Triton 风格）
│   ├── resnet34/
│   │   ├── 1/
│   │   │   └── model.onnx     # 模型文件
│   │   └── config.yaml         # 模型配置
│   ├── mnist_pretrained/
│   │   ├── 1/
│   │   │   └── model.onnx
│   │   └── config.yaml
│   └── resnet50/
│       ├── 1/
│       │   └── model.onnx
│       └── config.yaml
├── src/
│   └── inference_server/
│       ├── __init__.py
│       ├── main.py             # 入口：启动 HTTP + gRPC + 调度器
│       ├── config.py           # Pydantic 配置模型
│       ├── metrics.py          # Prometheus 指标定义
│       ├── model_manager.py    # 模型加载/生命周期管理
│       ├── api/                # 协议层
│       │   ├── http_server.py  # FastAPI HTTP 服务
│       │   ├── grpc_server.py  # gRPC 服务
│       │   ├── schemas.py      # 请求/响应 Pydantic 模型
│       │   ├── inference.proto # gRPC Protobuf 定义
│       │   ├── inference_pb2.py      # 生成的 Protobuf Python 代码
│       │   └── inference_pb2_grpc.py # 生成的 gRPC 代码
│       ├── core/               # 核心业务逻辑
│       │   ├── scheduler.py    # 动态批处理调度器
│       │   ├── preprocessor.py # 图像预处理
│       │   ├── postprocessor.py# 分类后处理
│       │   └── request.py      # 推理请求数据类
│       └── backend/            # 推理后端（与 serving 完全解耦）
│           ├── base.py         # Backend 抽象基类
│           ├── onnx_backend.py # ONNX Runtime 实现
│           ├── registry.py     # Backend 注册表
│           └── __init__.py
├── tests/                      # 单元测试
│   ├── test_scheduler.py       # 调度器测试
│   ├── test_backend.py         # Backend 测试
│   ├── test_preprocessor.py    # 预处理/后处理测试
│   ├── test_config.py          # 配置模型测试
│   └── test_integration.py     # HTTP 集成测试
├── scripts/                    # 工具脚本
│   ├── export_resnet34.py      # 导出 ResNet34 ONNX 模型
│   ├── export_mnist_onnx.py    # 用 numpy 创建 MNIST 模型
│   ├── download_mnist.py       # 下载 MNIST 数据集
│   ├── test_mnist_e2e.py       # MNIST 端到端测试
│   ├── test_resnet34.py        # ResNet34 推理测试
│   ├── test_static_batch_model.py  # 静态 batch 兼容性测试
│   ├── benchmark.py            # HTTP 压测工具
│   └── ...
└── data/                       # 数据集（MNIST 样本）
    └── mnist/
        └── test/
            └── {label}/
                └── {id}.png
```

---

## 已有模型速查

| 模型 | 任务 | 输入 | 输出 | 预处理 |
|------|------|------|------|--------|
| `resnet34` | ImageNet 图像分类 | RGB 224×224 | 1000 类 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| `mnist_pretrained` | MNIST 手写数字 | 灰度 28×28 | 10 类 | mean=[0.5], std=[0.5] |
| `resnet50` | 图像分类（测试用） | RGB 224×224 | 10 类 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

---

## 许可证

MIT License
