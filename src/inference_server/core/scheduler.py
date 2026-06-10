"""动态批处理调度器 — Pipeline 架构（收集与推理并行）。"""

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

    Pipeline 架构：
        请求 → request_queue → batch_collector(协程1) → batch_queue → batch_executor(协程2) → Future.set_result

    两个协程并行执行：
    - batch_collector: 持续凑 batch，不受推理阻塞
    - batch_executor: 持续处理 batch，不受收集阻塞

    消除串行架构下"推理期间请求在队列中干等"的问题。
    """

    def __init__(self, max_batch_size: int, max_wait_ms: float,
                 queue_capacity: int):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.queue = asyncio.Queue(maxsize=queue_capacity)
        self._shutdown = False

        # Pipeline 队列：连接收集器和执行器
        self._batch_queue = asyncio.Queue(maxsize=2)

        # 可观测性统计
        self._stats = {
            "total_batches": 0,
            "total_requests": 0,
            "dynamic_batches": 0,
            "single_batches": 0,
        }

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
        """启动 Pipeline：收集协程 + 推理协程并行运行。"""
        collector_task = asyncio.create_task(
            self._batch_collector(),
            name="batch_collector"
        )
        executor_task = asyncio.create_task(
            self._batch_executor(backend, preprocessor, postprocessor),
            name="batch_executor"
        )

        try:
            await asyncio.gather(collector_task, executor_task)
        except asyncio.CancelledError:
            pass
        finally:
            collector_task.cancel()
            executor_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass
            try:
                await executor_task
            except asyncio.CancelledError:
                pass

    async def _batch_collector(self) -> None:
        """收集协程：持续从 request_queue 凑 batch，放入 batch_queue。

        与推理协程并行运行，推理慢时 batch_queue 满会自然阻塞收集。
        """
        while not self._shutdown:
            batch = await self._collect_batch()
            if not batch:
                continue

            try:
                await self._batch_queue.put(batch)
            except asyncio.CancelledError:
                return

    async def _batch_executor(self, backend: Backend, preprocessor, postprocessor) -> None:
        """推理协程：持续从 batch_queue 取 batch，执行预处理→推理→后处理。

        与收集协程并行运行，处理期间收集协程继续工作。
        """
        while not self._shutdown:
            try:
                batch = await asyncio.wait_for(
                    self._batch_queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            self._stats["total_batches"] += 1
            self._stats["total_requests"] += len(batch)

            # ── 1. 预处理 ──
            pre_start = time.monotonic()
            try:
                images = [req.inputs.get("image") for req in batch]
                tensors = preprocessor.process_batch(images)
            except Exception as e:
                for req in batch:
                    req.set_exception(e)
                continue
            pre_latency = time.monotonic() - pre_start

            # ── 2. 构造输入 ──
            input_specs = backend.get_input_specs()
            input_name = input_specs[0]["name"] if input_specs else "input"
            inputs_list = [
                {
                    input_name: (
                        t[np.newaxis, ...]
                        if t.ndim == len(input_specs[0]["shape"]) - 1
                        else t
                    )
                }
                for t in tensors
            ]

            # ── 3. Backend 推理 ──
            infer_start = time.monotonic()
            try:
                results = backend.infer_batch(inputs_list)
            except Exception as e:
                for req in batch:
                    req.set_exception(e)
                continue
            infer_latency = time.monotonic() - infer_start

            # ── 4. 后处理 ──
            post_start = time.monotonic()
            for req, result_dict in zip(batch, results):
                try:
                    output_name = list(result_dict.keys())[0]
                    result = postprocessor.process(result_dict[output_name])
                    req.set_result(result)
                except Exception as e:
                    req.set_exception(e)
            post_latency = time.monotonic() - post_start

            # ── 5. 可观测性日志 ──
            is_dynamic = len(inputs_list) > 1 and getattr(
                backend, "_supports_dynamic_batch", False
            )
            strategy = "dynamic" if is_dynamic else "single"
            if is_dynamic:
                self._stats["dynamic_batches"] += 1
            else:
                self._stats["single_batches"] += 1

            # 队列堆积情况
            req_queue_depth = self.queue.qsize()
            batch_queue_depth = self._batch_queue.qsize()

            print(
                f"[batch] size={len(batch)}/{self.max_batch_size}, "
                f"strategy={strategy}, "
                f"pre={pre_latency * 1000:.1f}ms, "
                f"infer={infer_latency * 1000:.1f}ms, "
                f"post={post_latency * 1000:.1f}ms | "
                f"queues: req={req_queue_depth}, batch={batch_queue_depth} | "
                f"batches: total={self._stats['total_batches']}, "
                f"dynamic={self._stats['dynamic_batches']}"
            )

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

    def get_stats(self) -> dict[str, Any]:
        """返回调度器统计信息。"""
        total = self._stats["total_batches"]
        return {
            "total_batches": total,
            "total_requests": self._stats["total_requests"],
            "dynamic_ratio": (
                self._stats["dynamic_batches"] / total if total > 0 else 0.0
            ),
            "single_ratio": (
                self._stats["single_batches"] / total if total > 0 else 0.0
            ),
        }
