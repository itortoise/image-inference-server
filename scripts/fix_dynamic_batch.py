"""修改 ONNX 模型，支持动态 batch。只修改 Reshape 的 shape 和输入/输出的 batch 维度。"""
import onnx
from onnx import numpy_helper, TensorProto
from onnx.helper import make_tensor
import numpy as np

model_path = "models/mnist_pretrained/1/model_orig.onnx"
output_path = "models/mnist_pretrained/1/model.onnx"

model = onnx.load(model_path)

# 1. 修改输入 batch 维度为动态
for inp in model.graph.input:
    if inp.name == "Input3":
        dim = inp.type.tensor_type.shape.dim[0]
        dim.dim_param = "batch_size"
        print(f"  Input '{inp.name}': batch dim set to dynamic")

# 2. 修改输出 batch 维度为动态
for out in model.graph.output:
    if out.name == "Plus214_Output_0":
        dim = out.type.tensor_type.shape.dim[0]
        dim.dim_param = "batch_size"
        print(f"  Output '{out.name}': batch dim set to dynamic")

# 3. 修改 Reshape shape: [1, 256] -> [0, 256] (0 = copy from input dim)
# 找到 Pooling160_Output_0_reshape0_shape initializer 并修改
for i, init in enumerate(model.graph.initializer):
    if init.name == "Pooling160_Output_0_reshape0_shape":
        # 替换为 [0, 256]
        new_arr = np.array([0, 256], dtype=np.int64)
        new_init = numpy_helper.from_array(new_arr, name=init.name)
        # 删除旧的，添加新的
        del model.graph.initializer[i]
        model.graph.initializer.add().CopyFrom(new_init)
        print(f"  Reshape shape '{init.name}': [1, 256] -> [0, 256]")
        break

# 4. 运行 shape inference
model = onnx.shape_inference.infer_shapes(model)

onnx.save(model, output_path)
print(f"\n✅ Saved to {output_path}")

# 5. 验证
import onnxruntime as ort
sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
print("\nValidation:")
for bs in [1, 4, 8, 16, 32]:
    test = np.random.randn(bs, 1, 28, 28).astype(np.float32)
    out = sess.run(None, {"Input3": test})
    print(f"  batch={bs}: input {test.shape} -> output {out[0].shape}")

# 6. 用真实图片验证
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
X_test = mnist.data[60000:].astype(np.float32)
y_test = mnist.target[60000:].astype(np.int64)

# 测试 100 张
import random
correct = 0
for idx in random.sample(range(len(X_test)), 100):
    img = X_test[idx].reshape(1, 1, 28, 28) / 255.0  # normalize
    out = sess.run(None, {"Input3": img.astype(np.float32)})
    pred = np.argmax(out[0], axis=1)[0]
    if pred == y_test[idx]:
        correct += 1

print(f"\n🎯 Accuracy on 100 real MNIST images: {correct}% ({correct}/100)")
