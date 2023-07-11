import sys
sys.path.append('/data/home/hemantmishra/CrypTen/')
import crypten
import copy
import onnx
from crypten.nn import onnx_converter
onnx_model = onnx.load("./model_16.onnx")

crypten_model = onnx_converter._to_crypten(onnx_model)

crypten_model.pytorch_model = copy.deepcopy(pytorch_model)

# make sure training / eval setting is copied:
crypten_model.train(mode=pytorch_model.training)
print("Khela khatam!!")


print("Khela khatam!!!")