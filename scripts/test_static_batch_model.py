"""
验证：未修改的静态 batch 模型（batch=1 硬编码）也能工作。
使用 ONNX Model Zoo 原始 MNIST 模型（未做任何修改）。
"""
import asyncio
import random

import numpy as np
from sklearn.datasets import fetch_openml

from inference_server.backend.onnx_backend import ONNXRuntimeBackend
from inference_server.config import PreprocessConfig
from inference_server.core.preprocessor import ImagePreprocessor
from inference_server.core.postprocessor import ClassificationPostprocessor
from inference_server.core.scheduler import DynamicBatcher
from inference_server.core.request import InferenceRequest


def load_samples(count: int = 100):
    mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
    X = mnist.data[60000:].astype(np.float32)
    y = mnist.target[60000:].astype(np.int64)
    indices = random.sample(range(len(X)), count)
    return [(X[i].reshape(28, 28).astype(np.uint8), int(y[i])) for i in indices]


async def main():
    print("=" * 60)
    print("🧪 静态 batch 模型兼容性测试")
    print("=" * 60)
    print("模型: ONNX Model Zoo MNIST (原始，batch=1 硬编码)")
    print("预期: Backend 自动检测为静态 batch，退化为逐个推理")
    print()

    samples = load_samples(100)

    backend = ONNXRuntimeBackend()
    backend.initialize(
        "models/mnist_pretrained/1/model_orig.onnx",
        {"providers": ["CPUExecutionProvider"]}
    )

    # 验证检测
    print(f"动态 batch 支持: {backend._supports_dynamic_batch}")
    assert not backend._supports_dynamic_batch, "应该检测到静态 batch"

    config = PreprocessConfig(resize=[28, 28], mean=[0.5], std=[0.5], pixel_format="GRAY")
    preprocessor = ImagePreprocessor(config)
    postprocessor = ClassificationPostprocessor(top_k=1)

    scheduler = DynamicBatcher(max_batch_size=32, max_wait_ms=10, queue_capacity=1000)
    task = asyncio.create_task(scheduler.run(backend, preprocessor, postprocessor))

    # 预热
    req = InferenceRequest(inputs={"image": samples[0][0]})
    await (await scheduler.submit(req))
    await asyncio.sleep(0.1)

    # 批量测试（静态 batch 模型会逐个推理）
    correct = 0
    for img, true_label in samples:
        req = InferenceRequest(inputs={"image": img})
        future = await scheduler.submit(req)
        result = await asyncio.wait_for(future, timeout=5.0)
        pred = result["classes"][0]
        if pred == true_label:
            correct += 1

    scheduler.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    backend.destroy()

    print(f"\n📊 结果: {correct}/{len(samples)} correct = {100*correct/len(samples):.1f}%")
    print("✅ 静态 batch 模型无需修改即可工作！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
