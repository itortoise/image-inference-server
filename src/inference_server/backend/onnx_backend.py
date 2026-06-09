"""ONNX Runtime Backend 实现。"""

from typing import Dict, List

import numpy as np
import onnxruntime as ort

from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend


@register_backend("onnxruntime")
class ONNXRuntimeBackend(Backend):
    """ONNX Runtime 推理后端。

    通过 providers 配置支持不同硬件：
    - CPUExecutionProvider: CPU (默认)
    - CUDAExecutionProvider: NVIDIA GPU (需 onnxruntime-gpu)

    Batch 策略：
    - 自动检测模型是否支持动态 batch
    - 支持：合并为 [B, ...] 一次推理（性能最优）
    - 不支持：退化为逐个推理（兼容性好，服务端不修改模型）
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

        # 检测是否支持动态 batch（batch 维度是否为 None/字符串）
        self._supports_dynamic_batch = self._detect_dynamic_batch()

    def infer_single(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """单张推理。inputs shape 为 [1, ...]。"""
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(output_names, inputs)
        return dict(zip(output_names, outputs))

    def infer_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """批量推理。

        如果模型支持动态 batch：合并为 [B, ...] 一次推理。
        如果不支持动态 batch：退化为基类默认实现（逐个推理）。
        """
        if not self._supports_dynamic_batch or len(inputs_list) == 1:
            # 不支持动态 batch 或只有单张：逐个推理
            return super().infer_batch(inputs_list)

        # 支持动态 batch：合并推理
        input_name = self._input_specs[0]["name"]
        output_name = self._output_specs[0]["name"]

        # 合并输入: list[dict] -> dict[merged_tensor]
        merged_input = np.concatenate([inp[input_name] for inp in inputs_list], axis=0)

        # 一次推理
        outputs = self.infer_single({input_name: merged_input})
        output_tensor = outputs[output_name]

        # 拆分结果
        results = []
        for i in range(len(inputs_list)):
            single_output = output_tensor[i:i + 1]
            results.append({output_name: single_output})

        return results

    def get_input_specs(self) -> list[dict]:
        return self._input_specs

    def get_output_specs(self) -> list[dict]:
        return self._output_specs

    def destroy(self) -> None:
        del self.session
        self.session = None

    def _detect_dynamic_batch(self) -> bool:
        """检测模型是否支持动态 batch。

        检查第一个输入的 batch 维度（第 0 维）是否为动态：
        - 动态：值为 None 或字符串（如 "batch_size"）
        - 静态：值为具体的整数（如 1）
        """
        if not self._input_specs:
            return False

        batch_dim = self._input_specs[0]["shape"][0]
        # 动态维度标记为 -1（我们解析时把 None/字符串转为 -1）
        return batch_dim == -1

    @staticmethod
    def _extract_specs(io_info) -> list[dict]:
        """从 ONNX session 的输入/输出信息中提取规格。"""
        specs = []
        for info in io_info:
            shape = []
            for dim in info.shape:
                if dim is None or isinstance(dim, str):
                    shape.append(-1)  # 动态维度
                elif isinstance(dim, (int, float)):
                    shape.append(int(dim) if int(dim) > 0 else -1)
                else:
                    shape.append(-1)
            specs.append({
                "name": info.name,
                "dtype": info.type,
                "shape": shape,
            })
        return specs
