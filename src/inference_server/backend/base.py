"""Backend 抽象基类 - 所有推理后端必须继承此类。"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Backend(ABC):
    """推理后端抽象基类。

    Backend 只负责"张量进张量出"，不接触 HTTP/gRPC 或原始图像数据。

    关键设计：Backend 自己决定如何处理 batch。
    - 支持动态 batch 的后端可以在 infer_batch 中合并推理
    - 不支持动态 batch 的后端使用基类默认实现（逐个推理）
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
    def infer_single(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """单张推理。

        Args:
            inputs: {input_name: tensor}，shape 为 [1, ...]

        Returns:
            {output_name: tensor}，shape 为 [1, ...]
        """
        pass

    def infer_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """批量推理。

        默认实现：逐个调用 infer_single。
        子类可覆盖此方法以实现合并推理（如合并为 [B, ...] 后一次推理）。

        Args:
            inputs_list: [{input_name: tensor}, ...]，每个 tensor shape [1, ...]

        Returns:
            [{output_name: tensor}, ...]，每个 tensor shape [1, ...]
        """
        return [self.infer_single(inp) for inp in inputs_list]

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
