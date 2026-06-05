"""动态批处理调度器。"""

import asyncio
import time
from typing import Any, List

import numpy as np

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
        """主循环：持续凑 batch → 推理 → 分发。

        关键设计：Scheduler 只负责凑 batch 和预处理，不碰张量合并/拆分。
        Backend 自己决定如何处理 batch（支持动态 batch 则合并推理，不支持则逐个推理）。
        """
        while not self._shutdown:
            batch = await self._collect_batch()
            if not batch:
                continue

            # 并行预处理 → list[tensor]
            try:
                images = [req.inputs.get("image") for req in batch]
                tensors = preprocessor.process_batch(images)
            except Exception as e:
                for req in batch:
                    req.set_exception(e)
                continue

            # 构造 inputs_list: [{input_name: tensor}, ...]
            input_specs = backend.get_input_specs()
            input_name = input_specs[0]["name"] if input_specs else "input"
            inputs_list = [{input_name: t[np.newaxis, ...] if t.ndim == len(input_specs[0]["shape"]) - 1 else t} for t in tensors]

            # Backend 推理（Backend 自己决定合并还是逐张）
            infer_start = time.monotonic()
            try:
                results = backend.infer_batch(inputs_list)
            except Exception as e:
                for req in batch:
                    req.set_exception(e)
                continue
            infer_latency = time.monotonic() - infer_start

            # 逐个后处理并分发结果
            for req, result_dict in zip(batch, results):
                try:
                    output_name = list(result_dict.keys())[0]
                    result = postprocessor.process(result_dict[output_name])
                    req.set_result(result)
                except Exception as e:
                    req.set_exception(e)

    async def _collect_batch(self) -> List[InferenceRequest]:
        """凑 batch：首请求触发窗口，后续请求在窗口期内加入，超时必发。"""
        batch = []
        deadline = None

        while len(batch) < self.max_batch_size:
            if deadline is None:
                try:
                    req = await self.queue.get()
                except asyncio.CancelledError:
                    return []
                if self._shutdown:
                    return []
                batch.append(req)
                deadline = time.monotonic() + self.max_wait_ms
            else:
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
