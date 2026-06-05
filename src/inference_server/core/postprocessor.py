"""后处理模块 - 将模型原始输出转换为结构化结果。"""

from typing import Any

import numpy as np


class Postprocessor:
    """后处理基类。"""

    def process(self, output: np.ndarray) -> dict[str, Any]:
        """处理模型输出。

        Args:
            output: 模型输出张量，shape [1, ...]（单样本）

        Returns:
            结构化结果字典
        """
        raise NotImplementedError


class ClassificationPostprocessor(Postprocessor):
    """图像分类后处理：softmax + topk。"""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def process(self, output: np.ndarray) -> dict[str, Any]:
        """分类后处理。

        Args:
            output: logits，shape [1, num_classes]

        Returns:
            {"classes": [...], "scores": [...]}
        """
        # 去除 batch 维度
        logits = output.reshape(-1)

        # softmax
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        probs = exp_logits / np.sum(exp_logits)

        # topk
        topk_indices = np.argsort(probs)[::-1][:self.top_k]
        topk_scores = probs[topk_indices]

        return {
            "classes": topk_indices.tolist(),
            "scores": topk_scores.tolist(),
        }
