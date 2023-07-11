import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, adj):
        x = torch.cat([x, adj], dim=1)
        x = self.linear(x)
        x = F.relu(x)
        return x

# Create an instance of the SAGEConv module
input_size_gnn = 64
model = SAGEConv(in_channels=input_size_gnn, out_channels=1)

# Create some example input tensors
x = torch.randn(10, input_size_gnn)
adj = torch.randn(10, input_size_gnn)

# Export the model to ONNX
torch.onnx.export(model, (x, adj), "sageconv.onnx", opset_version=11)
