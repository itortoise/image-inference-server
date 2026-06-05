"""图像预处理模块。"""

from typing import List

import cv2
import numpy as np

from inference_server.config import PreprocessConfig


class ImagePreprocessor:
    """图像预处理器。

    执行以下操作：
    1. 格式转换 (BGR -> RGB 等)
    2. Resize
    3. 归一化 (mean/std)
    4. 维度转换 (HWC -> CHW)
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.target_size = tuple(config.resize)
        self.mean = np.array(config.mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(config.std, dtype=np.float32).reshape(3, 1, 1)
        self.pixel_format = config.pixel_format.upper()

    def process(self, image: np.ndarray) -> np.ndarray:
        """预处理单张图像。

        Args:
            image: numpy array, HWC format, dtype uint8

        Returns:
            numpy array, CHW format, dtype float32, normalized
        """
        # 确保是 uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 格式转换
        if self.pixel_format == "RGB" and image.shape[2] == 3:
            pass  # 保持原样
        elif self.pixel_format == "BGR" and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 转 float32 并归一化到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # 应用 mean/std
        result = (chw - self.mean) / self.std

        return result

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """批量预处理。"""
        return [self.process(img) for img in images]

    def merge_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        """将预处理后的单张图像张量合并成 batch 张量 [B, C, H, W]。"""
        return np.stack(tensors, axis=0)
