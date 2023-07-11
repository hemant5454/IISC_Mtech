import onnx
from onnx import helper, TensorProto

model = onnx.load("/data/home/hemantmishra/examples/CrypTen/model_onnx_new_1.onnx")

op_type = "LSTM"

clip_nodes = [node for node in model.graph.node if node.op_type == op_type]

for node in clip_nodes:
    node.input[4] = "sequence_lens"
## Pop the element
# clip_nodes[0].input[2] = "max_v"

Y = helper.make_tensor('sequence_lens', TensorProto.FLOAT, [], [0.])
model.graph.initializer.append(Y)

onnx.save(model, "model_onnx_new_2.onnx")