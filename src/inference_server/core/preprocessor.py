"""图像预处理模块。"""

from typing import List

import cv2
import numpy as np

from inference_server.config import PreprocessConfig


class ImagePreprocessor:
    """图像预处理器。

    支持 RGB(3通道) 和 GRAY(单通道) 输入。
    执行以下操作：
    1. 格式转换
    2. Resize
    3. 归一化 (mean/std)
    4. 维度转换 (HWC -> CHW 或 HW -> CHW)
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.target_size = tuple(config.resize)
        self.mean = np.array(config.mean, dtype=np.float32)
        self.std = np.array(config.std, dtype=np.float32)
        self.pixel_format = config.pixel_format.upper()
        self.num_channels = len(config.mean)  # 从 mean 推断通道数

    def process(self, image: np.ndarray) -> np.ndarray:
        """预处理单张图像。

        Args:
            image: numpy array, HWC format (彩色) 或 HW format (灰度), dtype uint8

        Returns:
            numpy array, CHW format, dtype float32, normalized
        """
        # 确保是 uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 处理单通道输入（可能是 HW 或 HWC with C=1）
        if len(image.shape) == 2:
            # HW -> HWC (1 channel)
            image = np.expand_dims(image, axis=-1)

        # 如果输入是 3 通道但模型期望 1 通道，转换为灰度
        if image.shape[2] == 3 and self.num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)

        # Resize
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 转 float32 并归一化到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # HWC -> CHW
        if len(normalized.shape) == 3:
            chw = np.transpose(normalized, (2, 0, 1))
        else:
            chw = np.expand_dims(normalized, axis=0)

        # 应用 mean/std (广播)
        if self.num_channels == 1:
            result = (chw - self.mean[0]) / self.std[0]
        else:
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)
            result = (chw - mean) / std

        return result

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """批量预处理。"""
        return [self.process(img) for img in images]

    def merge_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        """将预处理后的单张图像张量合并成 batch 张量 [B, C, H, W]。"""
        return np.stack(tensors, axis=0)
