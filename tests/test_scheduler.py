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
        # infer_batch 返回 list[dict]，每个 dict 包含单张输出
        backend.infer_batch = Mock(return_value=[
            {"output": np.ones((1, 10))},
            {"output": np.ones((1, 10))},
            {"output": np.ones((1, 10))},
            {"output": np.ones((1, 10))},
        ])
        backend.get_input_specs = Mock(return_value=[{"name": "images", "shape": [None, 3, 224, 224]}])
        return backend

    @pytest.fixture
    def mock_preprocessor(self):
        preprocessor = Mock()
        preprocessor.process_batch = Mock(return_value=[np.ones((3, 224, 224)), np.ones((3, 224, 224))])
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

        # 验证 backend.infer_batch 被调用，且传了 2 个输入
        mock_backend.infer_batch.assert_called_once()
        call_args = mock_backend.infer_batch.call_args[0][0]
        assert len(call_args) == 2  # batch size = 2

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
