import numpy as np
import pytest

from inference_server.config import PreprocessConfig
from inference_server.core.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    def test_preprocess_rgb_image(self):
        """测试 RGB 图像预处理"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            pixel_format="RGB",
        )
        preprocessor = ImagePreprocessor(config)

        # 模拟 RGB 图像: 400x300x3
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        result = preprocessor.process(image)

        assert result.shape == (3, 224, 224)  # NCHW
        assert result.dtype == np.float32

    def test_preprocess_with_normalization(self):
        """测试包含归一化的预处理"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            pixel_format="RGB",
        )
        preprocessor = ImagePreprocessor(config)

        # 全白图像 (255, 255, 255)
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        result = preprocessor.process(image)

        # 归一化后: (255/255 - 0.5) / 0.5 = 1.0
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_preprocess_bgr_to_rgb(self):
        """测试 BGR 转 RGB"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            pixel_format="BGR",
        )
        preprocessor = ImagePreprocessor(config)

        # BGR 图像: 蓝色通道=255
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # B = 255

        result = preprocessor.process(image)

        # 转换后 RGB: R=0, G=0, B=255/255=1.0
        assert result[0, 0, 0] == 0.0  # R
        assert result[1, 0, 0] == 0.0  # G
        assert result[2, 0, 0] == 1.0  # B

    def test_preprocess_batch(self):
        """测试批量预处理"""
        config = PreprocessConfig(
            resize=[224, 224],
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            pixel_format="RGB",
        )
        preprocessor = ImagePreprocessor(config)

        images = [
            np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),
            np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
        ]

        results = preprocessor.process_batch(images)

        assert len(results) == 2
        assert all(r.shape == (3, 224, 224) for r in results)


from inference_server.core.postprocessor import ClassificationPostprocessor


class TestClassificationPostprocessor:
    def test_softmax_topk(self):
        """测试分类后处理：softmax + topk"""
        postprocessor = ClassificationPostprocessor(top_k=3)

        # 模拟模型输出: [1, 5] - 5 个类别的 logits
        output = np.array([[2.0, 1.0, 0.5, 3.0, 0.1]], dtype=np.float32)
        result = postprocessor.process(output)

        assert "classes" in result
        assert "scores" in result
        assert len(result["classes"]) == 3
        assert len(result["scores"]) == 3

        # 最高分应该是索引 3
        assert result["classes"][0] == 3
        assert result["scores"][0] > result["scores"][1]
        assert all(0 <= s <= 1 for s in result["scores"])  # 每个分数在 [0,1] 之间
