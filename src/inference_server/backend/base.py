"""Backend 抽象基类 - 所有推理后端必须继承此类。"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Backend(ABC):
    """推理后端抽象基类。

    Backend 只负责"张量进张量出"，不接触 HTTP/gRPC 或原始图像数据。
    """

    @abstractmethod
    def initialize(self, model_path: str, config: Dict) -> None:
        """加载模型，准备推理环境。

        Args:
            model_path: 模型文件路径
            config: backend 特定配置（从 model config.yaml 的 backend_config 透传）
        """
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理。

        Args:
            inputs: {input_name: tensor}，张量已预处理好，shape 为 [B, ...]

        Returns:
            {output_name: tensor}，shape 为 [B, ...]
        """
        pass

    @abstractmethod
    def get_input_specs(self) -> list[dict]:
        """返回输入张量规格（name, dtype, shape）。"""
        pass

    @abstractmethod
    def get_output_specs(self) -> list[dict]:
        """返回输出张量规格。"""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """释放资源。"""
        pass
