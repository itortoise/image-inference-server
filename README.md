# Inference Server - 通用图像识别推理服务

一个支持**动态批处理**的通用图像识别推理服务，模型推理后端可配置、可替换，支持 HTTP/gRPC 双协议。

核心设计思想参考 [Triton Inference Server](https://github.com/triton-inference-server/server)：
- **配置驱动**：模型行为由 `config.yaml` 决定，修改配置即可切换后端、batch 策略
- **Backend 与 Serving 解耦**：新增推理引擎只需新增一个 backend 文件
- **动态批处理**：收集多个请求组成 batch，小模型在大显存机器上充分利用算力

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/macbook/workspace/personal_project/myzone/inference-server
pip install -e ".[dev]"
```

> 注意：`onnxruntime-gpu` 已包含在依赖中。如果目标机器有 NVIDIA GPU 且安装了 CUDA 12.x + cuDNN 9.x，会自动使用 GPU 推理；否则自动 fallback 到 CPU。

### 2. 启动服务

```bash
python -m inference_server.main --config configs/service.yaml
```

服务启动后：
- HTTP API: http://localhost:8000
- gRPC API: localhost:8001
- 健康检查: http://localhost:8000/v2/health/ready

### 3. 测试推理

```bash
# 查看已加载的模型
curl http://localhost:8000/v2/models

# 测试 ResNet34 图像分类
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

# 测试 MNIST 手写数字识别
curl -X POST http://localhost:8000/v2/models/mnist_pretrained/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["<base64编码的灰度图像>"]
    }]
  }'
```

---

## 项目结构

```
inference-server/
├── pyproject.toml              # 依赖管理
├── Dockerfile                  # Docker 打包
├── configs/
│   └── service.yaml            # 服务主配置（端口、batch策略、预加载模型）
├── models/                     # 模型仓库（参考 Triton model_repository）
│   ├── resnet34/               # 模型名称
│   │   ├── 1/                  # 版本号
│   │   │   └── model.onnx     # 模型文件
│   │   └── config.yaml         # 模型配置（backend、预处理、输入输出）
│   └── mnist_pretrained/       # 另一个模型
│       ├── 1/
│       │   └── model.onnx
│       └── config.yaml
├── src/
│   └── inference_server/       # 服务主包
│       ├── main.py             # 入口：启动 HTTP + gRPC + 调度器
│       ├── config.py           # Pydantic 配置模型
│       ├── api/                # 协议层
│       │   ├── http_server.py  # FastAPI HTTP 服务
│       │   ├── grpc_server.py  # gRPC 服务
│       │   └── schemas.py      # 请求/响应模型
│       ├── core/               # 核心业务逻辑
│       │   ├── scheduler.py    # 动态批处理调度器
│       │   ├── preprocessor.py # 图像预处理
│       │   ├── postprocessor.py# 分类后处理（softmax + topk）
│       │   └── request.py      # 推理请求数据类
│       ├── backend/            # 推理后端（与 serving 完全解耦）
│       │   ├── base.py         # Backend 抽象基类
│       │   ├── onnx_backend.py # ONNX Runtime 实现（含 GPU 支持）
│       │   └── registry.py     # Backend 注册表
│       └── model_manager.py    # 模型加载/生命周期管理
└── tests/                      # 测试
```

---

## 核心概念

### Backend 抽象

```
Client Request → HTTP/gRPC → Scheduler → Preprocessor → Backend → Postprocessor → Response
                                              ↑                ↑
                                              │                │
                                         图像resize/norm    ONNX Runtime / PyTorch / TensorRT
```

**Backend 只做"张量进张量出"**：
- `infer_single(inputs)`：单张推理
- `infer_batch(inputs_list)`：批量推理（默认逐个，子类可覆盖为合并推理）

**ONNXRuntimeBackend 自动适配**：
- 模型支持动态 batch → 合并为 `[B,C,H,W]` 一次推理（性能最优）
- 模型不支持动态 batch → 退化为逐个推理（兼容性好，不改模型）

### 配置分层

| 配置级别 | 文件 | 控制内容 |
|---------|------|---------|
| 服务级 | `configs/service.yaml` | 端口、调度器参数、预加载模型列表 |
| 模型级 | `models/{name}/config.yaml` | backend类型、输入输出shape、预处理参数、GPU provider |

---

## 添加自己的 ONNX 模型

### 步骤 1：创建目录

```bash
mkdir -p models/my_model/1
cp your_model.onnx models/my_model/1/model.onnx
```

### 步骤 2：写模型配置 `models/my_model/config.yaml`

```yaml
name: "my_model"
backend: "onnxruntime"
max_batch_size: 16

input:
  - name: "input"           # ONNX 模型里的输入节点名
    data_type: "FP32"
    dims: [3, 224, 224]    # 单张输入的 shape（不含 batch 维度）

output:
  - name: "output"          # ONNX 模型里的输出节点名
    data_type: "FP32"
    dims: [1000]

preprocess:
  resize: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"       # 或 "GRAY"

backend_config:
  providers:
    - name: "CUDAExecutionProvider"   # NVIDIA GPU
      options:
        device_id: 0
    - name: "CPUExecutionProvider"    # 兜底 fallback
```

> **怎么查 ONNX 模型的输入输出节点名？**
> ```python
> import onnxruntime as ort
> sess = ort.InferenceSession("model.onnx")
> print([i.name for i in sess.get_inputs()])   # 输入节点
> print([o.name for o in sess.get_outputs()])  # 输出节点
> ```

### 步骤 3：加到服务配置

编辑 `configs/service.yaml`：

```yaml
models:
  preload:
    - "my_model"
```

### 步骤 4：重启服务

```bash
python -m inference_server.main --config configs/service.yaml
```

---

## HTTP API 使用指南

### 健康检查

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/health/live
```

### 列出模型

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

### 推理请求

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

**图像编码方式**：
```python
import base64
import cv2

# 读取图像
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 默认 BGR，需转 RGB

# 编码为 PNG base64
_, buffer = cv2.imencode(".png", image)
image_base64 = base64.b64encode(buffer).decode("utf-8")
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
      "data": [{
        "classes": [973, 599, 539, 109, 107],
        "scores": [0.85, 0.12, 0.02, 0.005, 0.003]
      }]
    }
  ]
}
```

### 完整 Python 客户端示例

```python
import base64
import cv2
import requests

def infer_image(model_name: str, image_path: str):
    # 读取并编码图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", image)
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    # 发送请求
    resp = requests.post(
        f"http://localhost:8000/v2/models/{model_name}/infer",
        json={
            "inputs": [{
                "name": "input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_b64]
            }]
        }
    )
    return resp.json()

# 使用
result = infer_image("resnet34", "cat.jpg")
print(f"Top class: {result['outputs'][0]['data'][0]['classes'][0]}")
```

---

## gRPC API 使用指南

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

---

## GPU 支持

### 当前已支持的硬件

| 硬件 | Provider | 配置方式 |
|------|---------|---------|
| NVIDIA GPU | CUDAExecutionProvider | `providers: [{name: "CUDAExecutionProvider", options: {device_id: 0}}]` |
| NVIDIA GPU (TensorRT 加速) | TensorrtExecutionProvider | `providers: [{name: "TensorrtExecutionProvider"}]` |
| 昇腾 NPU | CANNExecutionProvider | `providers: [{name: "CANNExecutionProvider", options: {device_id: 0}}]` |
| CPU (兜底) | CPUExecutionProvider | `providers: [{name: "CPUExecutionProvider"}]` |

### 目标机器 GPU 环境检查

```python
import onnxruntime as ort
print(ort.get_available_providers())
# 有 GPU 时: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# 无 GPU 时: ['CPUExecutionProvider']
```

### CUDA 版本要求

`onnxruntime-gpu 1.26` 需要：
- CUDA 12.x
- cuDNN 9.x
- `libcublasLt.so.12` 在 `LD_LIBRARY_PATH`

---

## Docker 部署

```bash
docker build -t inference-server .
docker run -p 8000:8000 -p 8001:8001 inference-server
```

---

## 测试

```bash
# 单元测试
pytest tests/ -v

# ResNet34 推理测试
python scripts/test_resnet34.py

# MNIST 推理测试
python scripts/test_mnist_e2e.py --mode batch

# 静态 batch 模型兼容性测试
python scripts/test_static_batch_model.py
```

---

## 已有模型速查

| 模型 | 任务 | 输入 | 输出 | 预处理 |
|------|------|------|------|--------|
| `resnet34` | ImageNet 图像分类 | RGB 224x224 | 1000 类 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| `mnist_pretrained` | MNIST 手写数字 | 灰度 28x28 | 10 类 | mean=[0.5], std=[0.5] |
