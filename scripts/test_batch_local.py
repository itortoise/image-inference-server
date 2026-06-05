"""
本地批量推理测试 - 不启动 HTTP 服务，直接调用 Backend + Scheduler。
用于验证动态 batching 在本地 CPU 上的行为。
"""
import asyncio
import time
from inference_server.backend.onnx_backend import ONNXRuntimeBackend
from inference_server.core.preprocessor import ImagePreprocessor
from inference_server.core.postprocessor import ClassificationPostprocessor
from inference_server.core.scheduler import DynamicBatcher
from inference_server.core.request import InferenceRequest
from inference_server.config import PreprocessConfig
import numpy as np


def generate_mnist_image() -> np.ndarray:
    """生成随机 28x28 灰度图像。"""
    image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    for _ in range(np.random.randint(2, 5)):
        x, y = np.random.randint(0, 20), np.random.randint(0, 20)
        w, h = np.random.randint(3, 10), np.random.randint(3, 10)
        image[y:y+h, x:x+w] = np.random.randint(100, 256)
    return image


async def main():
    print("=" * 60)
    print("🧪 本地批量推理测试 (MNIST)")
    print("=" * 60)

    # 加载模型
    backend = ONNXRuntimeBackend()
    backend.initialize("models/mnist/1/model.onnx", {"providers": ["CPUExecutionProvider"]})
    print("✅ Model loaded")

    # 预处理/后处理
    preprocess_config = PreprocessConfig(
        resize=[28, 28],
        mean=[0.5],
        std=[0.5],
        pixel_format="GRAY",
    )
    preprocessor = ImagePreprocessor(preprocess_config)
    postprocessor = ClassificationPostprocessor(top_k=3)

    # 调度器
    scheduler = DynamicBatcher(
        max_batch_size=16,
        max_wait_ms=20,
        queue_capacity=1000,
    )

    # 启动调度器
    scheduler_task = asyncio.create_task(
        scheduler.run(backend, preprocessor, postprocessor)
    )

    # 预热
    print("🔥 预热...")
    req = InferenceRequest(inputs={"image": generate_mnist_image()})
    future = await scheduler.submit(req)
    await future
    await asyncio.sleep(0.1)

    # 批量测试
    total = 100
    concurrency = 20
    print(f"\n🚀 发送 {total} 个请求 (并发={concurrency})...")

    latencies = []
    start = time.monotonic()

    async def sender():
        for _ in range(total // concurrency):
            futures = []
            for _ in range(concurrency):
                req = InferenceRequest(inputs={"image": generate_mnist_image()})
                future = await scheduler.submit(req)
                futures.append((time.monotonic(), future))

            for t0, future in futures:
                result = await asyncio.wait_for(future, timeout=5.0)
                latencies.append(time.monotonic() - t0)

    await sender()

    total_time = time.monotonic() - start
    scheduler.shutdown()
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

    print(f"\n📊 结果:")
    print(f"  总请求:     {total}")
    print(f"  总耗时:     {total_time:.3f}s")
    print(f"  吞吐量:     {total/total_time:.2f} req/s")
    print(f"  平均延迟:   {np.mean(latencies)*1000:.2f}ms")
    print(f"  P50 延迟:   {np.percentile(latencies, 50)*1000:.2f}ms")
    print(f"  P99 延迟:   {np.percentile(latencies, 99)*1000:.2f}ms")
    print(f"  最小延迟:   {np.min(latencies)*1000:.2f}ms")
    print(f"  最大延迟:   {np.max(latencies)*1000:.2f}ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
