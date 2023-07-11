import torch
import torch.onnx
from torch import nn
from torch.nn import LSTM, Linear
from torch_geometric.nn import GCNConv
import torch_geometric

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.softmax(x, dim=-1)

# Create an instance of the LSTM model
input_size = 10
hidden_size = 32
num_layers = 2
lstm_model = LSTMModel(input_size, hidden_size, num_layers)

# Create an instance of the GNN model
num_classes = 2
gnn_model = GNNModel(input_size, hidden_size, num_classes)

# Generate random input tensors
batch_size = 1
sequence_length = 5
x_lstm = torch.empty(batch_size, sequence_length, input_size)
x_gnn = torch.empty(batch_size, input_size)
edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)

# Export the models to ONNX format
torch.onnx.export(lstm_model, x_lstm, "lstm_model_x.onnx", export_params=True)
torch.onnx.export(gnn_model, (x_gnn, edge_index), "gnn_model_x.onnx", export_params=True)
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

def conc_tensor(tensor1, tensor2, tensor3, tensor4):
    # Assuming the four tensors are named tensor1, tensor2, tensor3, and tensor4
    max_x = 34
    max_y = 60
    # Create an extra tensor to store the values of x and y
    # extra_tensor = torch.tensor([[tensor1.shape[1], tensor1.shape[2]]], dtype=torch.long)
    x_val = tensor1.shape[1]
    y_val = tensor1.shape[2]
    extra_tensor = torch.zeros((1, max_x, max_y), dtype=torch.long)

    # Assign the values of x and y to the appropriate elements
    extra_tensor[0, :1, :2] = torch.tensor([x_val, y_val], dtype=torch.long)
    if torch.cuda.is_available():
         extra_tensor = extra_tensor.cuda()
    # Pad the tensors to the maximum size
    tensor1 = F.pad(tensor1, (0, max_y-tensor1.shape[2], 0, max_x-tensor1.shape[1]))
    tensor2 = F.pad(tensor2, (0, max_y-tensor2.shape[2], 0, max_x-tensor2.shape[1]))
    tensor3 = F.pad(tensor3, (0, max_y-tensor3.shape[2], 0, max_x-tensor3.shape[1]))
    tensor4 = F.pad(tensor4, (0, max_y-tensor4.shape[2], 0, max_x-tensor4.shape[1]))
    # Concatenate the tensors into one tensor
    concat_tensor = torch.cat((extra_tensor, tensor1, tensor2, tensor3, tensor4), dim=0)
    return concat_tensor

def conc_tensor_2(tensor1):
    # Assuming the four tensors are named tensor1, tensor2, tensor3, and tensor4
    x = 17
    new_size = (18,)
    num_repeats = new_size[0] - len(tensor1)
    xx = torch.zeros(num_repeats, dtype=tensor1.dtype)
    if torch.cuda.is_available():
         xx = xx.cuda()
    last_value = len(tensor1)
    b = torch.cat((tensor1, xx), dim=0)
    b[-1] = last_value
    return b.unsqueeze(0)

def retrieving_tensors_2(tensor):
     tensor = tensor.squeeze(0)
     len = tensor[0,-1].item()
     var_sent = tensor[0][:len]
     var_act = tensor[1][:len]
     pass


# Slice the concatenated tensor to retrieve the extra tensor with x and y values
def retrieving_tensors(concat_tensor):
    max_x = 17
    max_y = 60
    
    orig_sizes = concat_tensor[0, :2].tolist()[0]

    # Slice the concatenated tensor to retrieve the individual tensors
    tensor1 = concat_tensor[1, :orig_sizes[0], :orig_sizes[1]]
    a = tensor1.unsqueeze(0)
    tensor2 = concat_tensor[2, :orig_sizes[0], :orig_sizes[0]]
    b = tensor2.unsqueeze(0)
    tensor3 = concat_tensor[3, :orig_sizes[0], :orig_sizes[0]]
    c = tensor3.unsqueeze(0)
    tensor4 = concat_tensor[4, :2*orig_sizes[0], :2*orig_sizes[0]]
    d = tensor4.unsqueeze(0)
    return a, b, c, d


#reading data
with open('/data/home/hemantmishra/examples/CrypTen/data/my_file.pkl', 'rb') as f:
        # Load the pickle data into a Python object
        data = pickle.load(f)
f.close()
# previous_tensor = torch.tensor([])
for i in range(len(data[0])):
    t1 = data[0][i]
    t2 = data[2][i]
    t3 = data[3][i]
    t4 = data[4][i]
    new_sent = data[5][i]
    new_act = data[6][i]
    concat_tensor = conc_tensor(t1, t2, t3, t4)
    new_tensor = concat_tensor.unsqueeze(0)
    new_sent = data[5][i]
    new_act = data[6][i]
    ext_sent = conc_tensor_2(new_sent)
    ext_act = conc_tensor_2(new_act)
    cur_label = torch.cat((ext_sent, ext_act), dim=0)
    cur_label = cur_label.unsqueeze(0)
    retrieving_tensors_2(cur_label)
    if(i==0):
         previous_tensor = new_tensor
         label = cur_label
         continue
    previous_tensor = torch.cat((previous_tensor, new_tensor), dim=0)
    label = torch.cat((label, cur_label), dim=0)
    

    # x, y, z, p = retrieving_tensors(concat_tensor)
       
with open('/data/home/hemantmishra/examples/CrypTen/data/four_tensor_con.pkl', 'wb') as f:
     pickle.dump(previous_tensor, f)
f.close()

with open('/data/home/hemantmishra/examples/CrypTen/data/two_label_con.pkl', 'wb') as f:
     pickle.dump(label, f)
f.close()

with open('/data/home/hemantmishra/examples/CrypTen/data/four_tensor_con.pkl', 'rb') as f:
        # Load the pickle data into a Python object
        data_x = pickle.load(f)
f.close()

with open('/data/home/hemantmishra/examples/CrypTen/data/two_label_con.pkl', 'rb') as f:
        # Load the pickle data into a Python object
        label_x = pickle.load(f)
f.close()

for i in tqdm(range(data_x.size(0))):
     # batch_loss = model.measure(*data_ba)
     x = data_x[i,:,:]
     y = label_x[i,:,:]
print('x')