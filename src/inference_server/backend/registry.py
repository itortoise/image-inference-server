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
