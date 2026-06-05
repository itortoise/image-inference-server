"""服务入口 - 启动 HTTP + gRPC + 调度器。"""

import asyncio
import argparse
from pathlib import Path

import uvicorn
import yaml

from inference_server.api.grpc_server import create_grpc_server
from inference_server.api.http_server import create_http_app
from inference_server.config import ServiceConfig
from inference_server.core.scheduler import DynamicBatcher
from inference_server.model_manager import ModelManager


async def serve(config: ServiceConfig):
    """启动服务。"""
    # 创建模型管理器
    model_manager = ModelManager(config.models.model_dir)

    # 加载模型
    if config.models.preload:
        model_manager.load_all(config.models.preload)
    else:
        model_manager.scan_and_load()

    # 创建调度器
    scheduler = DynamicBatcher.from_config(config.scheduler)

    # 获取第一个模型的 backend/preprocessor/postprocessor 用于调度器
    models = model_manager.list_models()
    if not models:
        raise RuntimeError("No models loaded")

    model_name = list(models.keys())[0]
    backend = model_manager.get_model_backend(model_name)
    preprocessor = model_manager.get_model_preprocessor(model_name)
    postprocessor = model_manager.get_model_postprocessor(model_name)

    # 启动调度器（后台协程）
    scheduler_task = asyncio.create_task(
        scheduler.run(backend, preprocessor, postprocessor)
    )

    # 创建 HTTP 服务
    http_app = create_http_app(model_manager, scheduler)

    # 创建 gRPC 服务
    grpc_server = create_grpc_server(
        model_manager, scheduler, config.server.grpc_port
    )

    # 启动 gRPC
    await grpc_server.start()
    print(f"gRPC server started on port {config.server.grpc_port}")

    # 启动 HTTP (uvicorn)
    config_uvicorn = uvicorn.Config(
        app=http_app,
        host="0.0.0.0",
        port=config.server.http_port,
        log_level="info",
    )
    server = uvicorn.Server(config_uvicorn)

    print(f"HTTP server starting on port {config.server.http_port}")
    print(f"Models loaded: {list(models.keys())}")

    try:
        await server.serve()
    except asyncio.CancelledError:
        pass
    finally:
        scheduler.shutdown()
        scheduler_task.cancel()
        await grpc_server.stop(5)


def main():
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/service.yaml",
        help="Path to service config file",
    )
    args = parser.parse_args()

    # 读取配置
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    config = ServiceConfig(**config_data)

    # 启动服务
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
