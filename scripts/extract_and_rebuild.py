"""
从 ONNX Model Zoo 的预训练 MNIST 模型提取权重，
重建一个支持动态 batch 的模型。
"""
import os

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_opsetid

# 加载预训练模型
pretrained = onnx.load("models/mnist_pretrained/1/model.onnx")

# 提取所有权重
weights = {}
for init in pretrained.graph.initializer:
    weights[init.name] = numpy_helper.to_array(init)
    print(f"  Weight: {init.name} -> shape={weights[init.name].shape}, dtype={weights[init.name].dtype}")

# 预训练模型结构 (opset 7, CNTK):
# Input3 [B,1,28,28] -> Conv(1,32,k=5) -> MaxPool(2) -> Conv(32,64,k=5) -> MaxPool(2)
# -> Flatten -> FC(1024,512) -> ReLU -> FC(512,10)

# 构建新图（支持动态 batch）
input_info = make_tensor_value_info("images", TensorProto.FLOAT, [None, 1, 28, 28])
output_info = make_tensor_value_info("output", TensorProto.FLOAT, [None, 10])

# 使用提取的权重创建 initializer
initializers = []
for name, arr in weights.items():
    initializers.append(numpy_helper.from_array(arr, name=name))

# 构建计算节点
nodes = [
    # Conv1: [B,1,28,28] -> [B,32,24,24]
    make_node("Conv", ["images", "Parameter5", "Parameter6"], ["Convolution28_Output_0"],
              dilations=[1,1], group=1, kernel_shape=[5,5], pads=[0,0,0,0], strides=[1,1]),
    # ReLU
    make_node("Relu", ["Convolution28_Output_0"], ["ReLU32_Output_0"]),
    # MaxPool: [B,32,24,24] -> [B,32,12,12]
    make_node("MaxPool", ["ReLU32_Output_0"], ["Pooling66_Output_0"],
              kernel_shape=[2,2], pads=[0,0,0,0], strides=[2,2]),
    
    # Conv2: [B,32,12,12] -> [B,64,8,8]
    make_node("Conv", ["Pooling66_Output_0", "Parameter87", "Parameter88"], ["Convolution110_Output_0"],
              dilations=[1,1], group=1, kernel_shape=[5,5], pads=[0,0,0,0], strides=[1,1]),
    # ReLU
    make_node("Relu", ["Convolution110_Output_0"], ["ReLU114_Output_0"]),
    # MaxPool: [B,64,8,8] -> [B,64,4,4]
    make_node("MaxPool", ["ReLU114_Output_0"], ["Pooling160_Output_0"],
              kernel_shape=[2,2], pads=[0,0,0,0], strides=[2,2]),
    
    # Flatten: [B,64,4,4] -> [B,1024]
    make_node("Flatten", ["Pooling160_Output_0"], ["Flatten_164_Output_0"], axis=1),
    
    # FC1: [B,1024] x [1024,512] -> [B,512]
    make_node("MatMul", ["Flatten_164_Output_0", "Parameter193"], ["Times212_Output_0"]),
    make_node("Add", ["Times212_Output_0", "Parameter194"], ["Plus215_Output_0"]),
    # ReLU
    make_node("Relu", ["Plus215_Output_0"], ["ReLU217_Output_0"]),
    
    # FC2: [B,512] x [512,10] -> [B,10]
    make_node("MatMul", ["ReLU217_Output_0", "Parameter253"], ["Times254_Output_0"]),
    make_node("Add", ["Times254_Output_0", "Parameter254"], ["Plus257_Output_0"]),
    
    # Softmax
    make_node("Softmax", ["Plus257_Output_0"], ["output"], axis=1),
]

graph = make_graph(
    nodes,
    "mnist_pretrained_dynamic",
    [input_info],
    [output_info],
    initializer=initializers,
)

model = make_model(graph, opset_imports=[make_opsetid("", 11)])
model.ir_version = 8

output_path = "models/mnist_pretrained/1/model.onnx"
onnx.save(model, output_path)

print(f"\n✅ Rebuilt model saved to {output_path}")

# 验证
import onnxruntime as ort
sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
for bs in [1, 4, 8, 16, 32]:
    test = np.random.randn(bs, 1, 28, 28).astype(np.float32)
    out = sess.run(None, {"images": test})
    print(f"  batch={bs}: input {test.shape} -> output {out[0].shape}")
