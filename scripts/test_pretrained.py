"""快速测试预训练 MNIST 模型。"""
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
    X = mnist.data[60000:].astype(np.float32)  # test set
    y = mnist.target[60000:].astype(np.int64)
    indices = random.sample(range(len(X)), count)
    return [(X[i].reshape(28, 28).astype(np.uint8), int(y[i])) for i in indices]


async def main():
    print("=" * 60)
    print("🧪 预训练 MNIST 模型测试")
    print("=" * 60)

    samples = load_samples(200)
    print(f"📦 Loaded {len(samples)} real MNIST images")

    backend = ONNXRuntimeBackend()
    backend.initialize(
        "models/mnist_pretrained/1/model.onnx",
        {"providers": ["CPUExecutionProvider"]}
    )
    print("✅ Pretrained model loaded")

    config = PreprocessConfig(resize=[28, 28], mean=[0.5], std=[0.5], pixel_format="GRAY")
    preprocessor = ImagePreprocessor(config)
    postprocessor = ClassificationPostprocessor(top_k=1)

    scheduler = DynamicBatcher(max_batch_size=32, max_wait_ms=10, queue_capacity=1000)
    task = asyncio.create_task(scheduler.run(backend, preprocessor, postprocessor))

    # warmup
    req = InferenceRequest(inputs={"image": samples[0][0]})
    await (await scheduler.submit(req))
    await asyncio.sleep(0.1)

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
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
