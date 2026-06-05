"""FastAPI HTTP 服务。"""

import asyncio
import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from inference_server.api.schemas import (
    ErrorResponse,
    InferRequest,
    InferResponse,
    ModelInfo,
    ModelListResponse,
)
from inference_server.config import ModelConfig
from inference_server.core.request import InferenceRequest
from inference_server.core.scheduler import DynamicBatcher


def create_http_app(
    model_manager,
    scheduler: DynamicBatcher,
) -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="Inference Server", version="0.1.0")

    @app.get("/v2/health/ready")
    async def health_ready():
        """服务就绪检查。"""
        return {"status": "ready"}

    @app.get("/v2/health/live")
    async def health_live():
        """服务存活检查。"""
        return {"status": "live"}

    @app.get("/v2/models", response_model=ModelListResponse)
    async def list_models():
        """列出所有模型。"""
        models = []
        for name, info in model_manager.list_models().items():
            models.append(ModelInfo(
                name=name,
                version="1",
                state=info["state"],
            ))
        return ModelListResponse(models=models)

    @app.post("/v2/models/{model_name}/infer", response_model=InferResponse)
    async def model_infer(model_name: str, request: InferRequest):
        """模型推理。"""
        # 获取模型配置
        model_config = model_manager.get_model_config(model_name)
        if model_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found",
            )

        # 解码图像
        try:
            input_data = request.inputs[0].data[0]
            if isinstance(input_data, str):
                # base64 编码的图像
                image_bytes = base64.b64decode(input_data)
            else:
                image_bytes = bytes(input_data)

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image data: {str(e)}",
            )

        # 创建推理请求
        req = InferenceRequest(
            model_name=model_name,
            inputs={"image": image},
        )

        # 提交到调度器
        try:
            future = await scheduler.submit(req)
            result = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference timeout",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

        # 构造响应
        return InferResponse(
            model_name=model_name,
            outputs=[
                {
                    "name": "output",
                    "shape": [1, len(result.get("classes", []))],
                    "datatype": "FP32",
                    "data": [result.get("classes", []), result.get("scores", [])],
                }
            ],
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """全局异常处理。"""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                }
            },
        )

    return app
