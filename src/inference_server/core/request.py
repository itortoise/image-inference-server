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
