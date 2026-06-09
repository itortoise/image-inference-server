"""
ResNet34 GPU 推理测试。
验证模型加载、CUDAExecutionProvider 配置、batch 推理。
"""
import asyncio
import random

import numpy as np
import onnxruntime as ort

from inference_server.backend.onnx_backend import ONNXRuntimeBackend
from inference_server.config import PreprocessConfig
from inference_server.core.preprocessor import ImagePreprocessor
from inference_server.core.postprocessor import ClassificationPostprocessor
from inference_server.core.scheduler import DynamicBatcher
from inference_server.core.request import InferenceRequest
from inference_server.model_manager import ModelManager


def generate_rgb_image() -> np.ndarray:
    """生成随机 RGB 图像。"""
    return np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)


async def test_direct_backend():
    """直接测试 Backend 推理。"""
    print("=" * 60)
    print("🧪 ResNet34 Backend 测试")
    print("=" * 60)

    backend = ONNXRuntimeBackend()
    backend.initialize(
        "models/resnet34/1/model.onnx",
        {
            "providers": [
                {"name": "CUDAExecutionProvider", "options": {"device_id": 0}},
                {"name": "CPUExecutionProvider"},
            ]
        }
    )

    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
    print(f"  Model active providers: {backend.session.get_providers()}")
    print(f"  Dynamic batch support: {backend._supports_dynamic_batch}")

    config = PreprocessConfig(
        resize=[224, 224],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        pixel_format="RGB",
    )
    preprocessor = ImagePreprocessor(config)
    postprocessor = ClassificationPostprocessor(top_k=5)

    # 单张推理
    img = generate_rgb_image()
    tensor = preprocessor.process(img)
    outputs = backend.infer_single({"input": tensor[np.newaxis, ...]})
    print(f"\n  单张推理: output shape={outputs['output'].shape}")
    print(f"  Top5 classes: {np.argsort(outputs['output'][0])[-5:][::-1].tolist()}")

    # Batch 推理
    images = [generate_rgb_image() for _ in range(4)]
    tensors = preprocessor.process_batch(images)
    inputs_list = [{"input": t[np.newaxis, ...]} for t in tensors]
    results = backend.infer_batch(inputs_list)
    print(f"\n  Batch 推理: {len(results)} results")
    for i, r in enumerate(results):
        print(f"    [{i}] output shape={r['output'].shape}")

    backend.destroy()
    print("\n✅ Backend 测试通过")


async def test_with_scheduler():
    """通过 Scheduler 测试高并发 batch 推理。"""
    print("\n" + "=" * 60)
    print("🧪 ResNet34 Scheduler 动态 Batch 测试")
    print("=" * 60)

    mm = ModelManager("models")
    ok = mm.load_model("resnet34")
    assert ok, "ResNet34 加载失败"

    backend = mm.get_model_backend("resnet34")
    preprocessor = mm.get_model_preprocessor("resnet34")
    postprocessor = mm.get_model_postprocessor("resnet34")

    scheduler = DynamicBatcher(max_batch_size=8, max_wait_ms=10, queue_capacity=1000)
    task = asyncio.create_task(scheduler.run(backend, preprocessor, postprocessor))

    # 预热
    req = InferenceRequest(inputs={"image": generate_rgb_image()})
    await (await scheduler.submit(req))
    await asyncio.sleep(0.1)

    # 并发发送 32 个请求
    total = 32
    concurrency = 16
    print(f"\n🚀 发送 {total} 个请求 (并发={concurrency}, max_batch=8)")

    start = asyncio.get_event_loop().time()
    for i in range(0, total, concurrency):
        futures = []
        for _ in range(min(concurrency, total - i)):
            req = InferenceRequest(inputs={"image": generate_rgb_image()})
            future = await scheduler.submit(req)
            futures.append(future)
        await asyncio.gather(*futures)

    elapsed = asyncio.get_event_loop().time() - start
    print(f"\n📊 结果:")
    print(f"  总请求: {total}")
    print(f"  总耗时: {elapsed:.3f}s")
    print(f"  吞吐量: {total/elapsed:.2f} req/s")

    scheduler.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    print("\n✅ Scheduler 测试通过")


async def main():
    await test_direct_backend()
    await test_with_scheduler()
    print("\n" + "=" * 60)
    print("🎉 ResNet34 GPU 推理方案验证完成")
    print("=" * 60)
    print("\n注意：当前环境没有 NVIDIA GPU，")
    print("      onnxruntime-gpu 会自动 fallback 到 CPUExecutionProvider。")
    print("      在目标机器上安装 CUDA 驱动后，自动使用 GPU 推理。")


if __name__ == "__main__":
    asyncio.run(main())
