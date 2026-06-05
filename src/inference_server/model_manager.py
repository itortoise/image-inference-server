"""模型管理器 - 负责模型加载、生命周期和配置。"""

import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from inference_server.backend.base import Backend
from inference_server.backend.registry import get_backend
from inference_server.backend import onnx_backend  # noqa: F401 - trigger registration
from inference_server.config import ModelConfig
from inference_server.core.postprocessor import ClassificationPostprocessor
from inference_server.core.preprocessor import ImagePreprocessor


class ModelInfo:
    """模型运行时信息。"""

    def __init__(self, config: ModelConfig, backend: Backend,
                 preprocessor: ImagePreprocessor, postprocessor):
        self.config = config
        self.backend = backend
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state = "READY"


class ModelManager:
    """模型管理器。

    负责：
    1. 从模型仓库扫描和加载模型
    2. 管理模型生命周期（LOADING → READY → UNLOADED）
    3. 提供模型配置查询
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._models: Dict[str, ModelInfo] = {}

    def load_model(self, model_name: str) -> bool:
        """加载单个模型。

        Returns:
            True if loaded successfully, False otherwise.
        """
        model_path = self.model_dir / model_name
        config_path = model_path / "config.yaml"

        if not config_path.exists():
            print(f"Model config not found: {config_path}")
            return False

        try:
            # 读取配置
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            model_config = ModelConfig(**config_data)

            # 查找模型文件
            version_dirs = sorted([d for d in model_path.iterdir() if d.is_dir()])
            if not version_dirs:
                print(f"No version directory found for model {model_name}")
                return False

            latest_version = version_dirs[-1]
            model_file = latest_version / "model.onnx"
            if not model_file.exists():
                # 尝试找其他格式的模型文件
                model_files = list(latest_version.glob("*.onnx"))
                if model_files:
                    model_file = model_files[0]
                else:
                    print(f"No model file found in {latest_version}")
                    return False

            # 创建 Backend
            backend_cls = get_backend(model_config.backend)
            backend = backend_cls()
            backend.initialize(str(model_file), model_config.backend_config)

            # 创建预处理器和后处理器
            preprocessor = ImagePreprocessor(model_config.preprocess)
            postprocessor = ClassificationPostprocessor(top_k=5)

            # 存储模型信息
            self._models[model_name] = ModelInfo(
                config=model_config,
                backend=backend,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

            print(f"Model '{model_name}' loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            return False

    def unload_model(self, model_name: str) -> None:
        """卸载模型。"""
        if model_name in self._models:
            self._models[model_name].backend.destroy()
            del self._models[model_name]
            print(f"Model '{model_name}' unloaded")

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置。"""
        info = self._models.get(model_name)
        return info.config if info else None

    def get_model_backend(self, model_name: str) -> Optional[Backend]:
        """获取模型 backend。"""
        info = self._models.get(model_name)
        return info.backend if info else None

    def get_model_preprocessor(self, model_name: str):
        """获取模型预处理器。"""
        info = self._models.get(model_name)
        return info.preprocessor if info else None

    def get_model_postprocessor(self, model_name: str):
        """获取模型后处理器。"""
        info = self._models.get(model_name)
        return info.postprocessor if info else None

    def list_models(self) -> Dict[str, dict]:
        """列出所有已加载的模型。"""
        return {
            name: {"state": info.state}
            for name, info in self._models.items()
        }

    def load_all(self, preload_list: list[str]) -> None:
        """加载指定的模型列表。"""
        for model_name in preload_list:
            self.load_model(model_name)

    def scan_and_load(self) -> None:
        """扫描模型目录并加载所有模型。"""
        if not self.model_dir.exists():
            return

        for item in self.model_dir.iterdir():
            if item.is_dir() and (item / "config.yaml").exists():
                self.load_model(item.name)
