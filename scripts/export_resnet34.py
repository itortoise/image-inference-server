"""导出 ResNet34 ONNX 模型（ImageNet 预训练权重）- 使用传统 export API。"""
import os

import torch
import torchvision


def export_resnet34(output_path: str = "models/resnet34/1/model.onnx"):
    """导出 ResNet34 ONNX 模型。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("⏳ Loading pretrained ResNet34...")
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"⏳ Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
        dynamo=False,  # 使用传统 export，确保权重完整保存
    )

    # 验证
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    test = torch.randn(2, 3, 224, 224).numpy()
    out = sess.run(None, {"input": test})
    print(f"✅ Exported: input {test.shape} -> output {out[0].shape}")
    print(f"   Model size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    export_resnet34()
