"""ONNX Runtime Backend 实现。"""

from typing import Dict

import numpy as np
import onnxruntime as ort

from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend


@register_backend("onnxruntime")
class ONNXRuntimeBackend(Backend):
    """ONNX Runtime 推理后端。

    通过 providers 配置支持不同硬件：
    - CANNExecutionProvider: 昇腾 NPU
    - CUDAExecutionProvider: NVIDIA GPU
    - CPUExecutionProvider: CPU (兜底)
    """

    def initialize(self, model_path: str, config: Dict) -> None:
        """加载 ONNX 模型。

        Args:
            model_path: ONNX 模型文件路径
            config: 必须包含 providers 列表
        """
        providers_config = config.get("providers", [
            {"name": "CPUExecutionProvider"}
        ])

        # 解析 providers 配置
        providers = []
        for p in providers_config:
            if isinstance(p, dict):
                providers.append((p["name"], p.get("options", {})))
            elif isinstance(p, str):
                providers.append(p)
            else:
                raise ValueError(f"Invalid provider config: {p}")

        self.session = ort.InferenceSession(model_path, providers=providers)

        # 缓存输入输出规格
        self._input_specs = self._extract_specs(self.session.get_inputs())
        self._output_specs = self._extract_specs(self.session.get_outputs())

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行 ONNX 推理。"""
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(output_names, inputs)
        return dict(zip(output_names, outputs))

    def get_input_specs(self) -> list[dict]:
        return self._input_specs

    def get_output_specs(self) -> list[dict]:
        return self._output_specs

    def destroy(self) -> None:
        del self.session
        self.session = None

    @staticmethod
    def _extract_specs(io_info) -> list[dict]:
        """从 ONNX session 的输入/输出信息中提取规格。"""
        specs = []
        for info in io_info:
            shape = []
            for dim in info.shape:
                if isinstance(dim, (int, float)):
                    shape.append(int(dim) if dim > 0 else -1)
                else:
                    shape.append(-1)
            specs.append({
                "name": info.name,
                "dtype": info.type,
                "shape": shape,
            })
        return specs
