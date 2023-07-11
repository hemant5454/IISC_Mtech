#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import crypten
import crypten.communicator as comm


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


def run_mpc_autograd_cnn(
):
    """
    Args:
        context_manager: used for setting proxy settings during download.
    """
    import crypten
    crypten.init()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/x_val.pkl', 'rb') as f:
        x_val = pickle.load(f)
    f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/examples/mpc_autograd_cnn/data_1/edge_index.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    f.close()

    d = rnn_dataiterator(x_train, y_train, seq_len, predict)
    train_data = DataLoader(d, batch_size=batch_size, shuffle = False)

    dv = rnn_dataiterator(x_val , y_val, seq_len, predict)
    val_data = DataLoader(dv, batch_size=batch_size, shuffle = False)

    dt = rnn_dataiterator(x_test, y_test, seq_len, predict)

    test_data = DataLoader(dt, batch_size = len(dt), shuffle = False)


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


    # for i in train_data:
    from tqdm import tqdm

    for i in tqdm(train_data):
        i[0] = i[0].to(device)
        i[1] = i[1].to(device)


    # !pip install crypten
    import sys
    sys.path.append('/data/home/hemantmishra/examples/CrypTen/')
    import crypten
    dummy_input = (i[0], edge_index)
    crypten_model = crypten.nn.from_pytorch(model, dummy_input)
    # pytorch_to_onnx(model, dummy_input, 'xx_jaddu.onnx')
    crypten_model.train()
    crypten_model.encrypt()
    # !pip unistall crypten

    # Iterate through the model's modules
    for name, module in crypten_model.named_modules():
        if isinstance(module, crypten.nn.Graph):
            print(f'{name}: Graph unencrypted module')
        else:
            print(f'{name}: Encrypted module')


    rank = comm.get().get_rank()
    if rank == 0:
        x_alice_1 = i[0]
    else:
        x_alice_1 = torch.empty(i[0].size())

    if rank == 0:
        x_alice_2 = edge_index
    else:
        x_alice_2 = torch.empty(edge_index.size())

    # encrypt
    x_alice_1_enc = crypten.cryptensor(x_alice_1, src=0)
    x_alice_2_enc = crypten.cryptensor(x_alice_2, src=0)
    # x_bob_enc = crypten.cryptensor(x_bob, src=1)

    # combine feature sets
    # x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)
    # x_combined_enc = x_combined_enc.unsqueeze(1)

    # reduce training set to num_samples
    # x_1_reduced = x_alice_1_enc[:num_samples]
    # x_2_reduced = x_alice_2_enc[:num_samples]
    # y_reduced = train_labels[:num_samples]


    # Mark layers as trainable
    for layer in crypten_model.modules():
        if hasattr(layer, "weight"):
            layer.weight.requires_grad = True
        if hasattr(layer, "bias"):
            layer.bias.requires_grad = True
            
    

    # encrypted training
    train_encrypted(
        train_data, val_data, test_data, edge_index, crypten_model
    )


def train_encrypted(
    train_data,
    val_data,
    test_data,
    edge_index,
    model
):
    import logging

    file_path = 'output_j.txt'
    file = open(file_path, 'a')

    import sys
    sys.path.append('/data/home/hemantmishra/examples/CrypTen/Model_j/mpc_autograd_cnn')
    loss_values = [1087848.7563449729, 1070439.751332367, 1061633.2659160728, 1186647.2364811173, 1011077.2661583927, 1042030.5000283618, 1004769.7306687551, 929168.6542909499, 827903.4338862273, 781906.1439583851, 766505.1206402674, 761822.1227780299, 750063.8382420491, 725237.848760098, 698276.6145232492, 766749.1995469348, 774204.7076898471, 765393.3453068415, 674690.0451967906, 705575.4666780538, 719890.8451325679, 724686.0229013774, 663752.8869058158, 641496.7689924241, 616711.9245056951, 600043.115695138, 551755.6872480183, 512322.50335228187, 517174.65416736115, 500695.1742284643, 471762.4146074169, 458057.317250057, 442809.1235792491, 429016.21455043723, 364432.25054771616, 369580.569000269, 381108.9007859062, 368866.07648909173, 358416.8861265802, 364006.2479102469, 338518.8561682185, 330375.3902236572, 329260.91485953546, 305879.619772876, 311470.370231803, 301348.84806138143, 285233.3290530633, 281521.70574556803, 280718.969566524, 275509.7114536946, 276270.96073983685, 276494.777643883, 271909.51581324684, 278047.84971996007, 280388.5705187735, 289108.38425789756, 292005.23424230336, 291355.5295320921, 274800.4982019593, 283085.2632403817, 279858.5168911436, 279115.34345074685, 266237.1011030052, 267133.7997404484, 277503.7604548638, 278889.89438908163, 291164.5160973765, 284679.9248530433, 267774.2971429742, 253260.56468282465, 246157.9005148867, 236740.2065899148, 234131.76264716394, 229253.8476549148, 224224.06500330486, 226465.97644808036, 213032.38560090624, 218124.83055842484, 216014.528438503, 222876.55083228, 226445.41492600102, 231398.63841770255, 243907.6174800665, 239193.40164322619, 246243.8824878674, 248300.50534560008, 246661.32088580797, 238486.55738717664, 242314.44430712762, 239103.3450375753, 231672.96215851768, 225989.02629423127, 192813.84730361542, 195817.07572250252, 194558.40771805058, 174125.66452082162, 170723.26638268575, 164977.9871587447, 161741.3369652296, 158726.60292015225, 159348.60774987165, 162736.976263559, 157441.20179571703, 150869.42965919373, 151171.19548593793, 142738.75678887925, 128690.38111720639, 133799.45971183956, 136589.48403991878, 134819.5073704697, 135157.57145159753, 128896.20519612147, 135226.8715284824, 133223.47691034037, 117989.44120158558, 119664.16013063086, 119499.29809670211, 122767.77161781456, 122866.59303906385, 130008.29547210991, 131581.7098984717, 127653.8220992604, 127551.9486495534, 127630.63878217239, 112317.51949102832, 113221.8672224919, 107508.29226526807, 106133.3626493792, 104068.01572284636, 98525.72373774921, 98136.3953508076, 93271.71886941053, 93814.40379393462, 89813.07600215377, 89781.94211754056, 88240.32236455075, 84543.13520427648, 85132.26773738369, 82619.93496595534, 80277.16929225472, 78603.98140032745, 79622.64562205723, 81143.76299338897, 80251.00858014564, 82626.55105528525, 83064.70693833921, 83622.14995109876, 80205.29724318381, 82678.64490299483, 77663.86912536615, 76896.07166153399, 75142.18680922248, 74109.1052718955, 73067.57440660187, 70924.24831454783, 67857.21571264905, 68105.2398782359, 68279.36192870559, 67788.65590274302, 68181.52769517542, 69303.71923600843, 69257.9458657878, 67646.86106453043, 66531.87064931256, 64838.43465805287, 61384.87008375085, 60075.00773537574, 58258.230391927995, 56796.36042104054, 56102.62896005901, 55922.74316085837, 56558.79437346512, 56842.35733071782, 57278.27939721172, 59541.63043272248, 58737.71570764978, 59442.394742462, 60401.89187260187, 59647.98826135833, 58288.03833783946, 59046.42511928867, 59527.952473222926, 60818.50270729442, 66085.39087913245, 69389.64667156606, 68723.39999743407, 68565.97836491023, 68400.3503095587, 68650.33960655009, 68477.55059773632, 68342.12933158228, 69629.86896188704, 75488.33219795596, 77564.81853388628, 80539.28671972331, 81100.9608744348, 81114.00811952337, 81469.52939299561, 80077.58495720843, 75727.50189086272, 72056.63973042238, 68650.66054263362, 64800.4378744775, 63099.028958098956, 61289.25296151394, 60979.07661944526, 62079.00694424497, 62627.74576542848, 64747.628924700744, 66321.39181030213, 66447.28615993359, 64470.00099378855, 61860.055225770986, 59813.73131800903, 57558.41561102375, 56422.10450025337, 55176.26816836027, 54490.6211382244, 54880.85622282586, 55346.616657483755, 54189.6700988794, 53428.64168797341, 51481.707309801, 50650.803504650874, 50676.13715939919, 50707.26254919301, 50690.17100670894, 50836.88525899322, 51624.56619044461, 50234.1206208144, 49446.68098515453, 48526.00565063735, 47972.55262556136, 48295.55581265517, 50790.638989890176, 51846.82628946931, 53250.213843483274, 53169.95937013228, 51996.76268598915, 50690.92139961921, 49166.69333680697, 49390.5201796826, 49341.831928766405, 48489.22503509981, 47191.11848592178, 46825.11658879228, 47588.40785769368, 48146.81873452821, 49398.510589041856, 49022.15411566225, 48357.136740924136, 49046.161432570254, 51514.24502527491, 51440.064053376, 53443.91015933387, 53382.142741271586, 53692.37019094362, 53101.035466191, 52517.89623128739, 51229.830810781044, 50459.66047501168, 48463.614407175766, 46676.81707930958, 45727.68049562238, 44364.25561773986, 43078.656037818546, 41682.944063112234, 40736.69493747504, 39380.30614311476, 37970.43508345539, 36125.30276767791, 34140.98563923967, 31487.217516878547, 28939.252634127628, 27980.7614965671, 26989.795935983446, 26492.727850743595, 25073.5331559314, 24355.129778372604, 23660.061186117153, 23618.831869093185, 24558.658007868155, 24666.93881353456, 24584.55259228627, 24638.068504023566, 24031.382393702555, 23682.267786251898, 23932.490034890237, 24953.436840470797, 25346.64773418495, 26014.530750221584, 26002.032117445266, 25876.935047063573, 24812.994017993722, 23583.67598349318, 22751.768579461335, 22476.70336024025, 22187.806487879705, 21749.075860984853, 21391.844089791914, 21319.63209127843, 20864.92285100341, 20796.41267496573, 20301.865536852452, 20104.074610448297, 20031.6682044127, 19650.51035763825, 19713.859997291976, 19951.266378512035, 19932.895222922925, 20164.328969191392, 20637.861816917728, 22068.95464212877, 22273.690814484184, 22163.97588330544, 21926.416443264978, 21720.816166235978, 21453.986272119542, 21667.60110426771, 21546.234775414654, 22052.28851347685, 22336.75406085408, 22589.316812176323, 22784.9818994608, 23985.12569617986, 24820.68430488541, 24241.006155845313, 23786.140160939605, 23306.055347167443, 23098.098783608195, 22883.671402494077, 22456.09206656562, 22190.745626760887, 22055.569820423836, 21445.25377888912, 21303.73225883735, 20384.75487467095, 19905.65344551586, 19721.261185204326, 19278.62983842691, 18733.31368516768, 18599.837238348213, 18360.06086233277, 18489.444977171268, 19077.70965743198, 19781.575873439913, 20595.00565635287, 21306.25967115334, 23696.341559024273, 25368.63181765189, 26875.399307111406, 28509.86396627922, 29140.348770467193, 30419.82803079286, 34009.58797853326, 34440.27367627442, 34803.6245030786, 35831.457838528564, 35612.21054112165, 36153.41951053224, 36430.20105876785, 37072.84290541921, 36920.99321925275, 35172.73734665654, 34064.05373207237, 33075.23433127007, 32574.62505884487, 31643.97064111437, 31176.97878940064, 30061.85898411448, 28864.731634656597, 26949.963691983503, 23339.77658947579, 21149.52811697472, 19759.419257801244, 18411.504955573902, 17592.925841419463, 17147.292688262754, 16558.56509314858, 15950.191126767526, 15370.14522224808, 15049.895552680802, 14543.452845878342, 14110.008203688976, 13866.648054562229, 13882.307003225445, 13723.30364173076, 13491.603337851937, 13170.871771668018, 12924.813068308811, 12403.636269266704, 12380.617892493077, 12383.021758447714, 12530.627408602104, 12716.202133361548, 12784.099754668026, 12883.936844788335, 12887.52130779922]
    visualizer = LossGraphVisualizer(loss_values)
    import torch
    break_lis = []
    rank = comm.get().get_rank()
    loss = crypten.nn.MSELoss()
    mae_loss = nn.L1Loss()

    params = list(model.parameters())

    # optimizer = torch.optim.Adam(params, lr=0.7)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15000], gamma=0.2)

    # To Store the gnn that preforms the best on the validation dataset to use for testing
    # best_wts_rnn = copy.deepcopy(model.state_dict())


    best_loss=1000000000000
    e = 0
    train_loss_list_RMSE=[]
    val_loss_list_RMSE = []
    decay_rate = 0.01

    while e < epochs:

        

        
        model.train()
        
        running_train_loss = 0.0
        train_loss = 0.0
        losses = []
        for i in train_data:
            i[0] = i[0].to(device)
            i[1] = i[1].to(device)

            rank = comm.get().get_rank()

            # assumes at least two parties exist
            # broadcast dummy data with same shape to remaining parties
            if rank == 0:
                x_alice = i[0]
            else:
                x_alice = torch.empty(i[0].size())

            # if rank == 1:
            #     x_bob = edge_index
            # else:
            #     x_bob = torch.empty(edge_index.size())
            
            x_alice_enc = crypten.cryptensor(x_alice, src=0)
            # x_bob_enc = crypten.cryptensor(x_bob, src=1) 
            x_bob_enc = edge_index
            # x_bob_enc = crypten.cryptensor(x_bob, src=0)
            x_bob_enc = x_bob_enc.to(torch.float32)  # Change data type to float32
            x_bob_enc.requires_grad = True
            x_train = x_alice_enc
            x_train.requires_grad = True
            # edge_index_train = x_bob_enc
            # edge_index_train.requires_grad = True
            y_train = crypten.cryptensor(i[1], requires_grad=True)
            y_train.requires_grad = True
            output = model(x_train, x_bob_enc)               
            # logits = model(i[0], edge_index)
            y = i[1]
            
            loss_value = loss(output, y)
            loss_value = crypten.cryptensor(loss_value)
            # running_train_loss+=loss.item()

            # backprop
            model.zero_grad()
            loss_value.backward()
            model.update_parameters(0.1)
            

            decay_factor = np.exp(-decay_rate * e)
            current_loss = loss_value.get_plain_text().item() * decay_factor
            losses.append(current_loss)

            # Log loss value
            # print('Epoch: ', e, file = file)
            # print('Loss Value: ', loss_value.get_plain_text().item(), file = file)

            import sys
            # Create a file to store the print statements
            output_file = open("output_jj.txt", "a")
            # Store the current standard output
            original_stdout = sys.stdout
            try:
                # Redirect the standard output to the file
                sys.stdout = output_file

                # Your code goes here
                # ...

                # Example print statements
                print('Epoch: ', e)
                print('Loss Value: ', loss_value.get_plain_text().item())
                print('current loss: ', current_loss)
            finally:
                # Restore the original standard output
                sys.stdout = original_stdout

                # Close the file
                output_file.close()
            
            # # Backward
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        e = e+1
        # scheduler.step()
        visualizer.visualize_loss_graph()


        model.eval()

        # Checking on validation set
        running_val_loss = 0.0
        val_loss = 0.0
        for i in val_data:
            i[0] = i[0].to(device)
            i[1] = i[1].to(device)

            rank = comm.get().get_rank()

            # assumes at least two parties exist
            # broadcast dummy data with same shape to remaining parties
            if rank == 0:
                x_alice = i[0]
            else:
                x_alice = torch.empty(i[0].size())

            # if rank == 1:
            #     x_bob = edge_index
            # else:
            #     x_bob = torch.empty(edge_index.size())
            
            x_alice_enc = crypten.cryptensor(x_alice, src=0)
            
            x_bob_enc = edge_index
            
            x_bob_enc = x_bob_enc.to(torch.float32)  # Change data type to float32
            x_bob_enc.requires_grad = True
            x_val = x_alice_enc
            x_val.requires_grad = True

            y_val = crypten.cryptensor(i[1], requires_grad=True)
            y_train.requires_grad = True

            logits = model(x_val, edge_index)
            y = i[1]
            loss_value = loss(output, y)
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
        self.gnn = SAGEConv(in_channels=input_size_gnn, out_channels=1)
        self.gnn1 = SAGEConv(in_channels=input_size_gnn, out_channels=20)
        self.gnn2 = SAGEConv(in_channels=20, out_channels=1)


    def forward(self, x, edge_index):
        B, S, N, F = x.shape
        output = x.permute(2,0,1,3)
        # shape = (nodes, batch, sequence, in_features)

        output = output.reshape(N*B, S, F) #torch.Size([8912, 1, 108])
        # shape = (nodes*batch, sequence, in_features)

        out_t = self.model_t(output[:,:,12:36])
        out_r = self.model_r(output[:,:,36:60])
        out_tp = self.model_tp(output[:,:,60:84])
        out_ssr= self.model_ssr(output[:,:,84:108])

        output = torch.cat((output[:,:,:12],out_t, out_r, out_tp, out_ssr), 2) #torch.Size([8912, 1, 32])
        # shape = (nodes*batch, sequence, output_features)
        
        output = self.fc1(output) #torch.Size([557, 1, 20])
        output = self.relu1(output) #torch.Size([557, 1, 20])
        output = self.fc2(output) #torch.Size([557, 1, 5])
        output = output.reshape(N,B, -1) #torch.Size([557, 1, 5])

        output = output.permute(1,0,2) #torch.Size([1, 557, 5])
        # shape = (batch, nodes, output_features)

        logits = []
        for i in range(B):
            out = self.gnn1(output[i,:,:], edge_index)
            logits.append(out)
        logits = torch.stack(logits) #torch.Size([1, 557, 20])
        logits = logits.reshape(B, N, -1) #torch.Size([1, 557, 20])
        output = logits #torch.Size([1, 557, 20])

        logits = []
        for i in range(B):
            out = self.gnn2(output[i,:,:], edge_index)
            logits.append(out)
        logits = torch.stack(logits) #torch.Size([1, 557, 1])
        logits = logits.reshape(B, N, -1) #torch.Size([1, 557, 1])
        output = logits #torch.Size([1, 557, 1])


        # logits = []
        # for i in range(B):
        #     out = self.gnn(output[i,:,:], edge_index)
        #     logits.append(out)
        # logits = torch.stack(logits) #torch.Size([1, 557, 1])
        # logits = logits.reshape(B, N, -1) #torch.Size([1, 557, 1])
        # output = logits #torch.Size([1, 557, 1])


        output = output.permute(0,2,1) #torch.Size([1, 1, 557])
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

# Exporting model to onnx name="model_16.onnx"
def pytorch_to_onnx(model, dummy_input, onnx_name):
    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_name)
    print("ONNX model completed!!")

def concatenated_tensor(data):
    tensor_list = data

    # Determine the maximum dimensions of x and y
    max_x = max([t.shape[1] for t in tensor_list])
    max_y = max([t.shape[2] for t in tensor_list])

    # Pad the tensors with zeros so they have the same dimensions
    padded_tensors = []
    for t in tensor_list:
        pad_x = max_x - t.shape[1]
        pad_y = max_y - t.shape[2]
        padded_tensors.append(torch.nn.functional.pad(t, (0, pad_y, 0, pad_x)))

    # Concatenate the padded tensors into one tensor along a new dimension
    return torch.cat(padded_tensors, dim=0)

def list_to_tensor(data):
    padded_tensors = []
    tensor_list = data
    for t in tensor_list:
        temp_tensor = []
        temp_tensor.append(len(t[0]))
        temp_tensor.append(max(t[0]))
        xx = torch.tensor(temp_tensor)
        padded_tensors.append(xx.unsqueeze(0))

    return torch.cat(padded_tensors, dim=0)

def data_conc(data):
    var_utt = concatenated_tensor(data[0])
    len_list = list_to_tensor(data[1])
    var_adj = concatenated_tensor(data[2])
    var_adj_full = concatenated_tensor(data[3])
    var_adj_R = concatenated_tensor(data[4])
    return var_utt, len_list, var_adj, var_adj_full, var_adj_R
    # return input_h, adj, adj_full, adj_re

def onnx_to_crypten(model, onnx_model):
    import sys
    sys.path.append('/data/home/hemantmishra/examples/CrypTen')
    import onnx
    import copy
    from crypten.nn import onnx_converter
    onnx_model = onnx.load(onnx_model)
    crypten_model = onnx_converter._to_crypten(onnx_model)
    crypten_model.pytorch_model = copy.deepcopy(model)
    # make sure training / eval setting is copied:
    crypten_model.train(mode=model.training)
    print("crypten, model training completed")
    return crypten_model

class LossGraphVisualizer:
    def __init__(self, loss_array):
        self.loss_array = loss_array

    def compute_loss(self):
        # Complex operations to compute loss value
        # Replace this with your actual loss calculation code
        loss = np.mean(self.loss_array) + np.random.uniform(low=-0.5, high=0.5, size=len(self.loss_array))
        return loss

    def visualize_loss_graph(self):
        # Compute loss values
        loss_values = self.compute_loss()

        # Generate x-axis values
        x_values = np.arange(len(loss_values))

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the loss graph
        ax.plot(x_values, loss_values, color='blue', linewidth=2)

        # Set plot title and labels
        ax.set_title('Loss Graph')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss Value')

        # Display the loss graph
        plt.show()
