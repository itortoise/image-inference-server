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
                 preprocessor: ImagePreprocessor, postprocessor, label_map: dict = None):
        self.config = config
        self.backend = backend
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.label_map = label_map or {}  # id -> key 映射
        self.state = "READY"


def _load_label_map(model_dir: Path, label_map_path: str) -> dict:
    """加载 char2idx.txt 文件，构建 id -> key 映射。

    文件格式（每行：key \t id）：
        cat    973
        dog    599
        bird   539

    Returns:
        {id: key} 字典
    """
    if not label_map_path:
        return {}

    # 支持相对路径（相对于模型目录）和绝对路径
    map_file = Path(label_map_path)
    if not map_file.is_absolute():
        map_file = model_dir / label_map_path

    if not map_file.exists():
        print(f"Label map file not found: {map_file}")
        return {}

    label_map = {}
    try:
        with open(map_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    try:
                        idx = int(parts[1])
                        label_map[idx] = key
                    except ValueError:
                        continue
        print(f"Loaded label map: {len(label_map)} entries from {map_file}")
    except Exception as e:
        print(f"Failed to load label map: {e}")

    return label_map


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

            # 加载 label 映射文件
            label_map = _load_label_map(model_path, model_config.label_map)

            # 创建预处理器和后处理器
            preprocessor = ImagePreprocessor(model_config.preprocess)
            postprocessor = ClassificationPostprocessor(top_k=5, label_map=label_map)

            # 存储模型信息
            self._models[model_name] = ModelInfo(
                config=model_config,
                backend=backend,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                label_map=label_map,
            )

            # 可观测性：打印模型加载摘要
            print(f"Model '{model_name}' loaded successfully")
            print(f"  ├─ Version:           {latest_version.name}")
            print(f"  ├─ Backend:           {model_config.backend}")
            print(f"  ├─ Label Map:         {len(label_map)} entries")
            print(f"  └─ Preprocess:        {model_config.preprocess.pixel_format}, "
                  f"channels={preprocessor.num_channels}, resize={preprocessor.target_size}")
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
