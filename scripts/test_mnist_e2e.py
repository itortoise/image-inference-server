"""
MNIST 端到端测试 - 使用真实数据集图片，验证完整流程。

完整链路:
    真实 PNG 图片 ──▶ 读取 (cv2.imread) ──▶ 编码 base64 ──▶ HTTP POST
        ──▶ 服务端解码 (base64→numpy) ──▶ 预处理 (resize/normalize/CHW)
        ──▶ Backend 推理 (ONNX Runtime) ──▶ 后处理 (softmax/topk)
        ──▶ 返回 JSON ──▶ 对比真实标签

模型: 使用 ONNX Model Zoo 预训练 MNIST 模型 (~94.5% 准确率)
数据: 70000 张真实手写数字图片
"""
import argparse
import asyncio
import base64
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_openml

from inference_server.backend.onnx_backend import ONNXRuntimeBackend
from inference_server.config import PreprocessConfig
from inference_server.core.preprocessor import ImagePreprocessor
from inference_server.core.postprocessor import ClassificationPostprocessor
from inference_server.core.scheduler import DynamicBatcher
from inference_server.core.request import InferenceRequest


def load_real_mnist(split: str = "test", count: int = 100) -> list[tuple[np.ndarray, int]]:
    """加载真实 MNIST 样本。优先用本地缓存，否则从 openml 下载。"""
    data_dir = Path(f"data/mnist/{split}")
    if data_dir.exists():
        all_files = []
        for label_dir in sorted(data_dir.iterdir()):
            if label_dir.is_dir():
                label = int(label_dir.name)
                for png_file in sorted(label_dir.glob("*.png")):
                    all_files.append((png_file, label))
        random.shuffle(all_files)
        selected = all_files[:count]
        return [(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), lbl) for f, lbl in selected]

    # 从 openml 下载
    print(f"  📥 Loading MNIST {split} from openml...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)
    if split == "train":
        images, labels = X[:60000], y[:60000]
    else:
        images, labels = X[60000:], y[60000:]
    indices = random.sample(range(len(images)), min(count, len(images)))
    return [(images[i].reshape(28, 28).astype(np.uint8), int(labels[i])) for i in indices]


def image_to_base64(image: np.ndarray) -> str:
    """numpy 图像 → base64 PNG。"""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(b64: str) -> np.ndarray:
    """base64 → numpy 灰度图像。"""
    buffer = base64.b64decode(b64)
    array = np.frombuffer(buffer, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)


async def test_single_pipeline(model_path: str = "models/mnist_pretrained/1/model.onnx"):
    """单张图片完整端到端流程验证。"""
    print("=" * 60)
    print("🧪 单张图片端到端流程测试")
    print("=" * 60)

    samples = load_real_mnist("test", count=1)
    image, true_label = samples[0]
    print(f"\n1️⃣  加载真实图片: shape={image.shape}, true_label={true_label}")

    # 2. 客户端编码
    b64 = image_to_base64(image)
    print(f"2️⃣  编码为 base64 PNG: {len(b64)} chars")

    # 3. 服务端解码
    decoded = base64_to_image(b64)
    print(f"3️⃣  服务端解码: shape={decoded.shape}, dtype={decoded.dtype}")

    # 4. 预处理
    config = PreprocessConfig(resize=[28, 28], mean=[0.5], std=[0.5], pixel_format="GRAY")
    preprocessor = ImagePreprocessor(config)
    tensor = preprocessor.process(decoded)
    print(f"4️⃣  预处理: shape={tensor.shape}")
    print(f"   min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}")

    # 5. Backend 推理
    backend = ONNXRuntimeBackend()
    backend.initialize(model_path, {"providers": ["CPUExecutionProvider"]})
    batch = preprocessor.merge_batch([tensor])
    outputs = backend.infer({"Input3": batch})
    print(f"5️⃣  Backend 推理: output shape={outputs['Plus214_Output_0'].shape}")

    # 6. 后处理
    postprocessor = ClassificationPostprocessor(top_k=3)
    result = postprocessor.process(outputs["Plus214_Output_0"])
    pred = result["classes"][0]
    match = "✅" if pred == true_label else "❌"
    print(f"6️⃣  后处理: top3={result['classes']}, scores={[f'{s:.4f}' for s in result['scores']]}")
    print(f"\n{match} 真实标签={true_label}, 预测标签={pred}")

    backend.destroy()
    return pred == true_label


async def test_batch_scheduler(model_path: str = "models/mnist_pretrained/1/model.onnx"):
    """调度器批量测试 - 真实图片 + 动态 batching。"""
    print("\n" + "=" * 60)
    print("🧪 调度器批量测试 (真实 MNIST + 动态 batching)")
    print("=" * 60)

    samples = load_real_mnist("test", count=500)
    print(f"\n📦 加载了 {len(samples)} 张真实 MNIST 图片")

    backend = ONNXRuntimeBackend()
    backend.initialize(model_path, {"providers": ["CPUExecutionProvider"]})

    config = PreprocessConfig(resize=[28, 28], mean=[0.5], std=[0.5], pixel_format="GRAY")
    preprocessor = ImagePreprocessor(config)
    postprocessor = ClassificationPostprocessor(top_k=1)

    scheduler = DynamicBatcher(max_batch_size=32, max_wait_ms=10, queue_capacity=1000)
    task = asyncio.create_task(scheduler.run(backend, preprocessor, postprocessor))

    # 预热
    req = InferenceRequest(inputs={"image": samples[0][0]})
    await (await scheduler.submit(req))
    await asyncio.sleep(0.1)

    total = len(samples)
    correct = 0
    latencies = []
    concurrency = 50

    print(f"\n🚀 发送 {total} 请求 (并发={concurrency}, max_batch=32, max_wait=10ms)")

    for i in range(0, total, concurrency):
        batch = samples[i:i+concurrency]
        futures = []
        t0 = time.monotonic()
        for img, _ in batch:
            req = InferenceRequest(inputs={"image": img})
            future = await scheduler.submit(req)
            futures.append(future)

        for j, (future, (_, true_label)) in enumerate(zip(futures, batch)):
            result = await asyncio.wait_for(future, timeout=5.0)
            pred = result["classes"][0]
            if pred == true_label:
                correct += 1

        latencies.append(time.monotonic() - t0)

    scheduler.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    backend.destroy()

    elapsed = sum(latencies)
    print(f"\n📊 结果:")
    print(f"  总请求:     {total}")
    print(f"  正确:       {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  总耗时:     {elapsed:.3f}s")
    print(f"  吞吐量:     {total/elapsed:.2f} req/s")
    print(f"  平均延迟:   {np.mean(latencies)*1000:.2f}ms")
    print(f"  P99 延迟:   {np.percentile(latencies, 99)*1000:.2f}ms")
    print("=" * 60)

    return correct / total


async def test_http_endpoint():
    """HTTP 端点测试（需要服务运行）。"""
    print("\n" + "=" * 60)
    print("🌐 HTTP 端点测试 (真实 MNIST 图片)")
    print("=" * 60)

    try:
        from httpx import AsyncClient
    except ImportError:
        print("  ⚠️ httpx not installed, skipping")
        return

    samples = load_real_mnist("test", count=50)
    print(f"\n📦 加载了 {len(samples)} 张真实图片")

    correct = 0
    latencies = []

    async with AsyncClient(base_url="http://localhost:8000") as client:
        for idx, (image, true_label) in enumerate(samples):
            b64 = image_to_base64(image)
            t0 = time.monotonic()
            try:
                resp = await client.post(
                    "/v2/models/mnist_pretrained/infer",
                    json={
                        "inputs": [
                            {
                                "name": "Input3",
                                "shape": [1],
                                "datatype": "BYTES",
                                "data": [b64],
                            }
                        ]
                    },
                    timeout=30.0,
                )
                latency = time.monotonic() - t0
                latencies.append(latency)
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data["outputs"][0]["data"][0][0]
                    ok = pred == true_label
                    if ok:
                        correct += 1
                    mark = "✅" if ok else "❌"
                    print(f"  {mark} [{idx+1:02d}] true={true_label} pred={pred}  ({latency*1000:.1f}ms)")
                else:
                    print(f"  ❌ [{idx+1:02d}] HTTP {resp.status_code}")
            except Exception as e:
                print(f"  ❌ [{idx+1:02d}] {e}")

    if latencies:
        print(f"\n📊 HTTP: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%) "
              f"avg={np.mean(latencies)*1000:.1f}ms")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="MNIST 端到端测试")
    parser.add_argument("--mode", choices=["single", "batch", "http", "all"], default="all")
    parser.add_argument("--model", default="models/mnist_pretrained/1/model.onnx", help="ONNX 模型路径")
    args = parser.parse_args()

    if args.mode in ("single", "all"):
        ok = await test_single_pipeline(args.model)
        if not ok:
            print("⚠️ 单张测试未通过（模型可能是随机权重）")

    if args.mode in ("batch", "all"):
        acc = await test_batch_scheduler(args.model)
        print(f"\n🎯 批量测试准确率: {acc*100:.1f}%")

    if args.mode in ("http", "all"):
        await test_http_endpoint()


if __name__ == "__main__":
    asyncio.run(main())
