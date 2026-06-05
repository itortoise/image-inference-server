"""
使用纯 onnx + numpy 创建 MNIST 测试模型（无需 PyTorch）。
网络结构: Conv(1,8) -> ReLU -> MaxPool -> Conv(8,16) -> ReLU -> MaxPool -> MatMul+Add(784,32) -> ReLU -> MatMul+Add(32,10)
使用 MatMul+Add 代替 Gemm 避免 ONNX Runtime 的融合优化问题。
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_opsetid

os.makedirs("models/mnist/1", exist_ok=True)

np.random.seed(42)

input_info = make_tensor_value_info("images", TensorProto.FLOAT, [None, 1, 28, 28])
output_info = make_tensor_value_info("output", TensorProto.FLOAT, [None, 10])

def make_tensor(name, shape, init_type='random'):
    if init_type == 'random':
        data = np.random.randn(*shape).astype(np.float32) * 0.1
    elif init_type == 'zeros':
        data = np.zeros(shape, dtype=np.float32)
    return numpy_helper.from_array(data, name=name)

# Conv1: 1 -> 8
W1 = make_tensor("W1", [8, 1, 3, 3])
B1 = make_tensor("B1", [8], 'zeros')

# Conv2: 8 -> 16
W2 = make_tensor("W2", [16, 8, 3, 3])
B2 = make_tensor("B2", [16], 'zeros')

# FC1 weights: [784, 32] (input features, output features)
W3 = make_tensor("W3", [784, 32])
B3 = make_tensor("B3", [32], 'zeros')

# FC2 weights: [32, 10]
W4 = make_tensor("W4", [32, 10])
B4 = make_tensor("B4", [10], 'zeros')

nodes = [
    # Conv1 + ReLU + Pool
    make_node("Conv", ["images", "W1", "B1"], ["c1"], pads=[1,1,1,1]),
    make_node("Relu", ["c1"], ["r1"]),
    make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2,2], strides=[2,2]),
    
    # Conv2 + ReLU + Pool
    make_node("Conv", ["p1", "W2", "B2"], ["c2"], pads=[1,1,1,1]),
    make_node("Relu", ["c2"], ["r2"]),
    make_node("MaxPool", ["r2"], ["p2"], kernel_shape=[2,2], strides=[2,2]),
    
    # Flatten: [B,16,7,7] -> [B,784]
    make_node("Flatten", ["p2"], ["flat"], axis=1),
    
    # FC1: MatMul([B,784], [784,32]) -> [B,32] + Add bias [32]
    make_node("MatMul", ["flat", "W3"], ["mm1"]),
    make_node("Add", ["mm1", "B3"], ["a1"]),
    make_node("Relu", ["a1"], ["r3"]),
    
    # FC2: MatMul([B,32], [32,10]) -> [B,10] + Add bias [10]
    make_node("MatMul", ["r3", "W4"], ["mm2"]),
    make_node("Add", ["mm2", "B4"], ["output"]),
]

graph = make_graph(
    nodes,
    "mnist_tiny",
    [input_info],
    [output_info],
    initializer=[W1, B1, W2, B2, W3, B3, W4, B4],
)

model = make_model(graph, opset_imports=[make_opsetid("", 11)])
model.ir_version = 8

with open("models/mnist/1/model.onnx", "wb") as f:
    f.write(model.SerializeToString())

print(f"✅ MNIST model saved: models/mnist/1/model.onnx")

# 验证
import onnxruntime as ort
sess = ort.InferenceSession("models/mnist/1/model.onnx", providers=["CPUExecutionProvider"])
for bs in [1, 2, 4, 8]:
    test_input = np.random.randn(bs, 1, 28, 28).astype(np.float32)
    outputs = sess.run(None, {"images": test_input})
    print(f"  batch={bs}: input {test_input.shape} -> output {outputs[0].shape}")

# 统计参数量
total_params = sum(np.prod(v.dims) for v in model.graph.initializer)
print(f"📊 Total parameters: {total_params:,} (~{total_params/1000:.1f}K)")
