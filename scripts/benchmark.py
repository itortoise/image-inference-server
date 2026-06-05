"""
MNIST 推理服务批量压测脚本。

用法:
    # 先启动服务
    python -m inference_server.main --config configs/service.yaml

    # 再运行压测
    python scripts/benchmark.py --concurrency 10 --total 100 --model mnist

功能:
    - 模拟并发 HTTP 请求
    - 测量吞吐量 (req/s)、P50/P99 延迟
    - 观察动态 batching 效果
    - 生成随机 MNIST 风格图像（无需下载数据集）
"""
import argparse
import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from httpx import AsyncClient


@dataclass
class BenchmarkResult:
    """压测结果。"""
    total_requests: int
    concurrency: int
    total_time: float
    successful: int = 0
    failed: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.successful / self.total_time if self.total_time > 0 else 0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0
        return np.percentile(self.latencies, 50)

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        return np.percentile(self.latencies, 99)

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0
        return np.mean(self.latencies)

    def report(self):
        print("=" * 60)
        print("📊 压测结果")
        print("=" * 60)
        print(f"  总请求数:     {self.total_requests}")
        print(f"  并发数:       {self.concurrency}")
        print(f"  成功:         {self.successful}")
        print(f"  失败:         {self.failed}")
        print(f"  总耗时:       {self.total_time:.3f}s")
        print(f"  吞吐量:       {self.throughput:.2f} req/s")
        print(f"  平均延迟:     {self.avg_latency*1000:.2f}ms")
        print(f"  P50 延迟:     {self.p50_latency*1000:.2f}ms")
        print(f"  P99 延迟:     {self.p99_latency*1000:.2f}ms")
        print("=" * 60)


def generate_mnist_image() -> str:
    """生成随机 MNIST 风格的 28x28 灰度图像，返回 base64。"""
    # 生成随机灰度图像 + 一些噪声模拟手写数字
    image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    # 添加一些结构（模拟笔画）
    for _ in range(np.random.randint(2, 5)):
        x, y = np.random.randint(0, 20), np.random.randint(0, 20)
        w, h = np.random.randint(3, 10), np.random.randint(3, 10)
        image[y:y+h, x:x+w] = np.random.randint(100, 256)

    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


async def send_request(client: AsyncClient, model_name: str, image_base64: str) -> float:
    """发送单个推理请求，返回延迟（秒）。"""
    start = time.monotonic()
    try:
        response = await client.post(
            f"/v2/models/{model_name}/infer",
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
            timeout=30.0,
        )
        elapsed = time.monotonic() - start
        if response.status_code == 200:
            return elapsed
        else:
            print(f"  ❌ Request failed: {response.status_code} - {response.text}")
            return -1
    except Exception as e:
        print(f"  ❌ Request error: {e}")
        return -1


async def worker(client: AsyncClient, model_name: str, queue: asyncio.Queue,
                 result: BenchmarkResult, semaphore: asyncio.Semaphore):
    """工作协程：从队列取请求并发送。"""
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        async with semaphore:
            latency = await send_request(client, model_name, item)
            if latency >= 0:
                result.latencies.append(latency)
                result.successful += 1
            else:
                result.failed += 1


async def run_benchmark(base_url: str, model_name: str, concurrency: int,
                        total_requests: int) -> BenchmarkResult:
    """运行压测。"""
    result = BenchmarkResult(
        total_requests=total_requests,
        concurrency=concurrency,
        total_time=0,
    )

    # 预生成所有图像
    print(f"🎲 生成 {total_requests} 张测试图像...")
    images = [generate_mnist_image() for _ in range(total_requests)]

    # 填充请求队列
    queue = asyncio.Queue()
    for img in images:
        await queue.put(img)

    # 并发控制
    semaphore = asyncio.Semaphore(concurrency)

    print(f"🚀 开始压测: concurrency={concurrency}, total={total_requests}")
    print(f"   目标: {base_url}, model: {model_name}")

    async with AsyncClient(base_url=base_url) as client:
        # 先预热一次
        print("🔥 预热...")
        await send_request(client, model_name, images[0])
        await asyncio.sleep(0.5)

        start_time = time.monotonic()

        # 启动所有 worker
        workers = [
            asyncio.create_task(worker(client, model_name, queue, result, semaphore))
            for _ in range(concurrency)
        ]

        await asyncio.gather(*workers)

        result.total_time = time.monotonic() - start_time

    return result


def main():
    parser = argparse.ArgumentParser(description="MNIST 推理服务压测")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--model", default="mnist", help="模型名称")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="并发数")
    parser.add_argument("--total", "-n", type=int, default=100, help="总请求数")
    parser.add_argument("--warmup", type=int, default=5, help="预热请求数")
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(
        base_url=args.url,
        model_name=args.model,
        concurrency=args.concurrency,
        total_requests=args.total,
    ))

    result.report()


if __name__ == "__main__":
    main()
