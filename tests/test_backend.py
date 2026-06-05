import pytest
from inference_server.backend.base import Backend
from inference_server.backend.registry import register_backend, get_backend


class TestBackendBase:
    def test_backend_is_abstract(self):
        """Backend 基类不能直接实例化"""
        with pytest.raises(TypeError):
            Backend()


class TestBackendRegistry:
    def test_register_and_get_backend(self):
        """测试注册和获取 backend"""

        @register_backend("test_backend")
        class TestBackend(Backend):
            def initialize(self, model_path, config):
                pass

            def infer(self, inputs):
                return inputs

            def get_input_specs(self):
                return []

            def get_output_specs(self):
                return []

            def destroy(self):
                pass

        cls = get_backend("test_backend")
        assert cls.__name__ == "TestBackend"

    def test_get_unknown_backend_raises(self):
        """获取未注册的 backend 应该报错"""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

import numpy as np
import tempfile
import os


class TestONNXRuntimeBackend:
    def _create_test_onnx_model(self) -> str:
        """创建一个简单的 ONNX 模型用于测试：y = x * 2 + 1"""
        try:
            import onnx
            from onnx import numpy_helper, TensorProto
            from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
        except ImportError:
            pytest.skip("onnx package not installed")

        # 输入: x [B, 3, 224, 224]
        # 输出: y [B, 3, 224, 224] = x * 2 + 1
        input_info = make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 224, 224])
        output_info = make_tensor_value_info("y", TensorProto.FLOAT, [None, 3, 224, 224])

        # 创建常量: scale=2, bias=1
        scale = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="scale")
        bias = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="bias")

        node_mul = make_node("Mul", ["x", "scale"], ["mul_out"])
        node_add = make_node("Add", ["mul_out", "bias"], ["y"])

        graph = make_graph(
            [node_mul, node_add],
            "test_model",
            [input_info],
            [output_info],
            initializer=[scale, bias],
        )
        model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)])
        model.ir_version = 8

        # 保存到临时文件
        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.write(fd, model.SerializeToString())
        os.close(fd)
        return path

    def test_onnx_backend_initialize_and_infer(self):
        """测试 ONNX Backend 初始化和推理"""
        from inference_server.backend.onnx_backend import ONNXRuntimeBackend

        model_path = self._create_test_onnx_model()
        backend = ONNXRuntimeBackend()
        backend.initialize(model_path, {"providers": ["CPUExecutionProvider"]})

        # 检查输入输出规格
        input_specs = backend.get_input_specs()
        assert len(input_specs) == 1
        assert input_specs[0]["name"] == "x"

        output_specs = backend.get_output_specs()
        assert len(output_specs) == 1
        assert output_specs[0]["name"] == "y"

        # 推理
        x = np.ones((2, 3, 224, 224), dtype=np.float32)
        outputs = backend.infer({"x": x})

        assert "y" in outputs
        expected = x * 2 + 1
        np.testing.assert_allclose(outputs["y"], expected, rtol=1e-5)

        backend.destroy()
        os.unlink(model_path)

    def test_onnx_backend_providers_config(self):
        """测试 providers 配置解析"""
        from inference_server.backend.onnx_backend import ONNXRuntimeBackend

        model_path = self._create_test_onnx_model()
        backend = ONNXRuntimeBackend()

        # 带 options 的 provider 配置
        config = {
            "providers": [
                {"name": "CPUExecutionProvider", "options": {}},
            ]
        }
        backend.initialize(model_path, config)
        assert backend.session is not None

        backend.destroy()
        os.unlink(model_path)
