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

            # 获取模型输入名（从 backend 的输入规格中获取第一个）
            input_specs = backend.get_input_specs()
            input_name = input_specs[0]["name"] if input_specs else "input"

            # 获取模型输出名
            output_specs = backend.get_output_specs()
            output_name = output_specs[0]["name"] if output_specs else "output"

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
            output_tensor = outputs[output_name]

            for i, req in enumerate(batch):
                try:
                    single_output = output_tensor[i:i + 1]
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
