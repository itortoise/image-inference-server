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
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from httpx import AsyncClient


@dataclass
class RequestResult:
    """单次请求结果。"""
    request_id: int
    latency: float  # 秒，-1 表示失败
    status_code: int = 0
    classes: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return self.latency >= 0 and self.status_code == 200


@dataclass
class BenchmarkResult:
    """压测结果。"""
    total_requests: int
    concurrency: int
    total_time: float
    successful: int = 0
    failed: int = 0
    latencies: List[float] = field(default_factory=list)
    request_results: List[RequestResult] = field(default_factory=list)

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

    def report(self, sample_size: int = 10):
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

        # 打印推理结果样本
        success_results = [r for r in self.request_results if r.success]
        if success_results:
            show_n = min(sample_size, len(success_results))
            print(f"\n🔍 推理结果样本（前 {show_n} 条成功请求）:")
            print("-" * 60)
            for r in success_results[:show_n]:
                top_class = r.classes[0] if r.classes else "N/A"
                top_score = r.scores[0] if r.scores else "N/A"
                score_str = f"{top_score:.4f}" if isinstance(top_score, float) else str(top_score)
                print(f"  req#{r.request_id:04d}  "
                      f"latency={r.latency*1000:6.2f}ms  "
                      f"top_class={top_class}  "
                      f"top_score={score_str}  "
                      f"classes={r.classes[:5]}")
            print("-" * 60)

        # 统计推理结果分布
        if success_results:
            all_top_classes = [
                r.classes[0] for r in success_results if r.classes
            ]
            if all_top_classes:
                unique, counts = np.unique(all_top_classes, return_counts=True)
                print(f"\n📈 Top-1 预测类别分布（共 {len(all_top_classes)} 条）:")
                for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
                    print(f"    class {cls}: {cnt} 次 ({cnt/len(all_top_classes)*100:.1f}%)")

        # 打印失败请求详情
        failed_results = [r for r in self.request_results if not r.success]
        if failed_results:
            print(f"\n❌ 失败请求详情（共 {len(failed_results)} 条）:")
            for r in failed_results[:5]:
                print(f"    req#{r.request_id:04d}: {r.error}")


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


async def send_request(client: AsyncClient, model_name: str,
                       request_id: int, image_base64: str) -> RequestResult:
    """发送单个推理请求，返回完整结果（含推理输出）。"""
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
            body = response.json()
            # 响应结构: {"model_name": ..., "outputs": [{"name": "output", "data": [classes, scores]}]}
            outputs = body.get("outputs", [])
            classes: list = []
            scores: list = []
            if outputs:
                data = outputs[0].get("data", [])
                if len(data) >= 2:
                    classes = data[0]
                    scores = data[1]
                elif len(data) == 1:
                    # 兼容只有 classes 的情况
                    classes = data[0]
            return RequestResult(
                request_id=request_id,
                latency=elapsed,
                status_code=200,
                classes=classes,
                scores=scores,
            )
        else:
            return RequestResult(
                request_id=request_id,
                latency=-1,
                status_code=response.status_code,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )
    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            request_id=request_id,
            latency=-1,
            error=f"{type(e).__name__}: {e}",
        )


async def worker(client: AsyncClient, model_name: str, queue: asyncio.Queue,
                 result: BenchmarkResult, semaphore: asyncio.Semaphore):
    """工作协程：从队列取 (request_id, image_base64) 并发送。"""
    while True:
        try:
            request_id, image_base64 = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        async with semaphore:
            req_result = await send_request(
                client, model_name, request_id, image_base64
            )
            result.request_results.append(req_result)
            if req_result.success:
                result.latencies.append(req_result.latency)
                result.successful += 1
            else:
                result.failed += 1
                print(f"  ❌ req#{req_result.request_id} 失败: {req_result.error}")


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

    # 填充请求队列（带 request_id）
    queue = asyncio.Queue()
    for idx, img in enumerate(images):
        await queue.put((idx, img))

    # 并发控制
    semaphore = asyncio.Semaphore(concurrency)

    print(f"🚀 开始压测: concurrency={concurrency}, total={total_requests}")
    print(f"   目标: {base_url}, model: {model_name}")

    async with AsyncClient(base_url=base_url) as client:
        # 先预热一次
        print("🔥 预热...")
        warmup_result = await send_request(client, model_name, -1, images[0])
        if warmup_result.success:
            print(f"   预热成功: latency={warmup_result.latency*1000:.2f}ms, "
                  f"classes={warmup_result.classes[:3]}, scores={warmup_result.scores[:3]}")
        else:
            print(f"   预热失败: {warmup_result.error}")
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
    parser.add_argument("--output", "-o", default="", help="结果保存路径（JSON），为空则不保存")
    parser.add_argument("--sample", type=int, default=10, help="报告中展示的推理结果样本数")
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(
        base_url=args.url,
        model_name=args.model,
        concurrency=args.concurrency,
        total_requests=args.total,
    ))

    result.report(sample_size=args.sample)

    # 保存完整结果到 JSON
    output_path = args.output
    if not output_path:
        output_path = f"benchmark_{args.model}_c{args.concurrency}_n{args.total}.json"

    save_data = {
        "config": {
            "url": args.url,
            "model": args.model,
            "concurrency": args.concurrency,
            "total_requests": args.total,
        },
        "summary": {
            "successful": result.successful,
            "failed": result.failed,
            "total_time": round(result.total_time, 4),
            "throughput": round(result.throughput, 2),
            "avg_latency_ms": round(result.avg_latency * 1000, 2),
            "p50_latency_ms": round(result.p50_latency * 1000, 2),
            "p99_latency_ms": round(result.p99_latency * 1000, 2),
        },
        "requests": [
            {
                "id": r.request_id,
                "success": r.success,
                "status_code": r.status_code,
                "latency_ms": round(r.latency * 1000, 3) if r.latency >= 0 else -1,
                "classes": r.classes,
                "scores": r.scores,
                "error": r.error,
            }
            for r in sorted(result.request_results, key=lambda x: x.request_id)
        ],
    }

    Path(output_path).write_text(json.dumps(save_data, ensure_ascii=False, indent=2))
    print(f"\n💾 完整结果已保存: {output_path}")


if __name__ == "__main__":
    main()
