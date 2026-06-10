import base64

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from inference_server.api.http_server import create_http_app
from inference_server.config import SchedulerConfig
from inference_server.core.scheduler import DynamicBatcher
from inference_server.model_manager import ModelManager


class TestIntegration:
    @pytest_asyncio.fixture
    async def http_client(self):
        """创建测试用的 HTTP 客户端。"""
        # 需要先有测试模型被加载
        model_manager = ModelManager("models")
        model_manager.load_model("resnet50")

        scheduler = DynamicBatcher.from_config(
            SchedulerConfig(max_batch_size=4, max_wait_ms=50, queue_capacity=100)
        )

        # 启动调度器
        import asyncio
        model_name = list(model_manager.list_models().keys())[0]
        backend = model_manager.get_model_backend(model_name)
        preprocessor = model_manager.get_model_preprocessor(model_name)
        postprocessor = model_manager.get_model_postprocessor(model_name)

        task = asyncio.create_task(
            scheduler.run(backend, preprocessor, postprocessor)
        )

        app = create_http_app(model_manager, scheduler)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        scheduler.shutdown()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client):
        """测试健康检查端点"""
        response = await http_client.get("/v2/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
        assert data["result"]["code"] == "200"

    @pytest.mark.asyncio
    async def test_list_models(self, http_client):
        """测试模型列表"""
        response = await http_client.get("/v2/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
        models = data["result"]["data"]["message"]["data"]
        assert any(m["name"] == "resnet50" for m in models)

    @pytest.mark.asyncio
    async def test_infer_with_base64_image(self, http_client):
        """测试推理端点"""
        # 创建测试图像
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        response = await http_client.post(
            "/v2/models/resnet50/infer",
            json={
                "inputs": [
                    {
                        "name": "images",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [image_base64],
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
        assert data["result"]["code"] == "200"
        assert data["result"]["data"]["message"]["data"] is not None
