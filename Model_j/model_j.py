import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from statistics import mean
from sklearn.preprocessing import MinMaxScaler

import geopy
import geopy.distance

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv

device = torch.device("cpu")
import pickle
predict = 30

# How many days of previous data to be considered, >1 only for RNN
seq_len = 1

# Moving average days for price
moving_average_days = 7

# Hyperparameters
epochs = 600
batch_size = 1
out_channels = 5
input_size_to_FC_Layer = 5


# Threshold Distance between 2 markets
distance_threshold = 200.0

with open('Model_j/data_1/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
f.close()
with open('Model_j/data_1/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
f.close()
with open('Model_j/data_1/x_val.pkl', 'rb') as f:
    x_val = pickle.load(f)
f.close()
with open('Model_j/data_1/y_val.pkl', 'rb') as f:
    y_val = pickle.load(f)
f.close()
with open('Model_j/data_1/x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
f.close()
with open('Model_j/data_1/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
f.close()
with open('Model_j/data_1/edge_index.pkl', 'rb') as f:
    edge_index = pickle.load(f)
f.close()

class rnn_dataiterator(Dataset):
    def __init__(self, input, y, seq_len, predict):
        self.input = input
        self.seq_len = seq_len
        self.predict = predict
        
        #As the output of the first sequence will only be the price at seq_len+1, the y array is sliced accordingly
        self.y=y[seq_len:]
        
    def __getitem__(self, item):
        # input sequence length is item - item+seq_len, and output length is item - item+predict length
        return self.input[item:item + self.seq_len], self.y[item + self.predict-1:item + self.predict]
    
    def __len__(self):
        return len(self.y) - self.predict + 1
    

d = rnn_dataiterator(x_train, y_train, seq_len, predict)
train_data = DataLoader(d, batch_size=batch_size, shuffle = False)

dv = rnn_dataiterator(x_val , y_val, seq_len, predict)
val_data = DataLoader(dv, batch_size=batch_size, shuffle = False)

dt = rnn_dataiterator(x_test, y_test, seq_len, predict)
test_data = DataLoader(dt, batch_size = len(dt), shuffle = False)



class CNN(torch.nn.Module):
    def __init__(self, out_channel):
        super(CNN, self).__init__()

        self.out_channel = out_channel
        self.c1   = torch.nn.Conv1d(in_channels=1, out_channels=self.out_channel, kernel_size=24, stride=1)

        self.relu = torch.nn.ReLU()
        self.avg  = torch.nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        x = x.float()
        out = self.c1(x)
        out = self.relu(out)
        
        out = out.permute(0, 2, 1)
        return out
    
# LSTM model
# Takes input of shape = [batch, sequence, input_features] and has an output of shape [batch, out_sequence, out_features]
class CNN_FC_GraphSAGE(nn.Module):

    def __init__(self, input_size_gnn = 5, layers_gnn = 16, out_channel = 5):
        super(CNN_FC_GraphSAGE, self).__init__()
        
        self.out_channel = out_channel

        self.model_t = CNN(out_channel)
        self.model_r = CNN(out_channel)
        self.model_tp = CNN(out_channel)
        self.model_ssr= CNN(out_channel)
        
        self.fc1 = nn.Linear(12+4*self.out_channel, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20,input_size_gnn)

        self.gnn1 = SAGEConv(in_channels=input_size_gnn, out_channels=20)
        self.gnn2 = SAGEConv(in_channels=20, out_channels=1)


    def forward(self, x, edge_index):
        B, S, N, F = x.shape
        output = x.permute(2,0,1,3)
        # shape = (nodes, batch, sequence, in_features)

        output = output.reshape(N*B, S, F)
        # shape = (nodes*batch, sequence, in_features)

        out_t = self.model_t(output[:,:,12:36])
        out_r = self.model_r(output[:,:,36:60])
        out_tp = self.model_tp(output[:,:,60:84])
        out_ssr= self.model_ssr(output[:,:,84:108])

        output = torch.cat((output[:,:,:12],out_t, out_r, out_tp, out_ssr), 2)
        # shape = (nodes*batch, sequence, output_features)
        
        output = self.fc1(output)
        output = self.relu1(output)
        output = self.fc2(output)
        output = output.reshape(N,B, -1)

        output = output.permute(1,0,2)
        # shape = (batch, nodes, output_features)

        logits = []
        for i in range(B):
            out = self.gnn1(output[i,:,:], edge_index)
            logits.append(out)
        logits = torch.stack(logits)
        logits = logits.reshape(B, N, -1)
        output = logits

        logits = []
        for i in range(B):
            out = self.gnn2(output[i,:,:], edge_index)
            logits.append(out)
        logits = torch.stack(logits)
        logits = logits.reshape(B, N, -1)
        output = logits

        output = output.permute(0,2,1)
        # shape = (batch, out_features, nodes)        
        return output
    
#  Functions for R2 loss and persons coefficient
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    
    
    r2 = 1 - ss_res / ss_tot
    return r2

def pearsons(x,y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost

model = CNN_FC_GraphSAGE().to(device)

Lmse = nn.MSELoss()
mae_loss = nn.L1Loss()

params = list(model.parameters())

optimizer = torch.optim.Adam(params, lr=0.7)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15000], gamma=0.2)

# To Store the gnn that preforms the best on the validation dataset to use for testing
best_wts_rnn = copy.deepcopy(model.state_dict())


best_loss=1000000000000
e = 0
train_loss_list_RMSE=[]
val_loss_list_RMSE = []


break_lis = []

while e < epochs:
    
    model.train()
    
    running_train_loss = 0.0
    train_loss = 0.0
    for i in train_data:
        i[0] = i[0].to(device)
        i[1] = i[1].to(device)
        
        logits = model(i[0], edge_index)
        y = i[1]
        
        loss = Lmse(logits, y)
        running_train_loss+=loss.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # scheduler.step()
    train_loss= running_train_loss/len(train_data)
    train_loss_list_RMSE.append(train_loss**0.5)


    model.eval()

    # Checking on validation set
    running_val_loss = 0.0
    val_loss = 0.0
    for i in val_data:
        i[0] = i[0].to(device)
        i[1] = i[1].to(device)

        logits = model(i[0], edge_index)
        y = i[1]
        loss = Lmse(logits, y)
        running_val_loss+=loss.item()

    val_loss= running_val_loss/len(val_data)
    val_loss_list_RMSE.append(val_loss**0.5)

    if val_loss <= best_loss:
        best_loss = val_loss
        best_wts_rnn = copy.deepcopy(model.state_dict())

    if len(break_lis)==5:
        break_lis = break_lis[1:]
    break_lis.append(val_loss)
    count = 0
    for ele in break_lis:
        if ele==val_loss:
            count = count+1
    if count==5:
        break

    print('In epoch {}, train loss: {:.3f} RMSE, val loss: {:.3f} RMSE'.format(e, train_loss**0.5, val_loss**0.5))
    
    e = e+1
