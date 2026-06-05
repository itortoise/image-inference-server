"""gRPC 服务实现。"""

import asyncio

import grpc

from inference_server.api import inference_pb2, inference_pb2_grpc
from inference_server.core.request import InferenceRequest


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC 服务实现。"""

    def __init__(self, model_manager, scheduler):
        self.model_manager = model_manager
        self.scheduler = scheduler

    async def ModelInfer(self, request, context):
        """模型推理。"""
        import cv2
        import numpy as np

        model_name = request.model_name
        model_config = self.model_manager.get_model_config(model_name)
        if model_config is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{model_name}' not found")
            return inference_pb2.ModelInferResponse()

        # 解码图像
        try:
            image_bytes = request.inputs[0].raw_data
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid image data: {str(e)}")
            return inference_pb2.ModelInferResponse()

        # 创建推理请求
        req = InferenceRequest(
            model_name=model_name,
            inputs={"image": image},
        )

        # 提交到调度器
        try:
            future = await self.scheduler.submit(req)
            result = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Inference timeout")
            return inference_pb2.ModelInferResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.ModelInferResponse()

        # 构造响应
        import json
        response_data = json.dumps(result).encode("utf-8")
        return inference_pb2.ModelInferResponse(
            model_name=model_name,
            outputs=[
                inference_pb2.InferOutput(
                    name="output",
                    shape=[1, len(result.get("classes", []))],
                    datatype="BYTES",
                    raw_data=response_data,
                )
            ],
        )

    async def ModelReady(self, request, context):
        """模型就绪检查。"""
        model_name = request.model_name
        model_config = self.model_manager.get_model_config(model_name)
        ready = model_config is not None
        return inference_pb2.ModelReadyResponse(ready=ready)


def create_grpc_server(model_manager, scheduler, port: int):
    """创建 gRPC 服务器。"""
    server = grpc.aio.server()
    servicer = InferenceServicer(model_manager, scheduler)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
