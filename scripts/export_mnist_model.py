"""导出 MNIST ONNX 模型 - 使用纯 PyTorch (无预训练，仅导出一个随机权重的小网络用于测试)。"""

import os
import torch
import torch.nn as nn


class TinyMNISTNet(nn.Module):
    """极简 MNIST CNN，参数量 ~8K"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [B, 8, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # [B, 16, 7, 7]
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def export_mnist_model(output_path: str = "models/mnist/1/model.onnx"):
    """导出 MNIST ONNX 模型。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = TinyMNISTNet()
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

    print(f"MNIST model exported to {output_path}")

    # 验证模型
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    test_input = torch.randn(4, 1, 28, 28).numpy()
    outputs = sess.run(None, {"images": test_input})
    print(f"Test inference passed: input {test_input.shape} -> output {outputs[0].shape}")


if __name__ == "__main__":
    export_mnist_model()
