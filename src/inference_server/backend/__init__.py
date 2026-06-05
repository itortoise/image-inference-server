"""Backend 模块 - 导入所有 backend 实现以触发注册。"""

from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend, get_backend, list_backends

# 导入所有 backend 实现以触发 @register_backend 装饰器
from inference_server.backend.onnx_backend import ONNXRuntimeBackend

__all__ = ["Backend", "register_backend", "get_backend", "list_backends", "ONNXRuntimeBackend"]
