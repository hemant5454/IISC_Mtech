#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import sys
sys.path.append('/data/home/hemantmishra/examples/CrypTen')
sys.path.append('/data/home/hemantmishra/examples/CrypTen/Co_GAT_1')

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.util import NoopContextManager
from torchvision import datasets, transforms

from utils import DataHub
import crypten
from nn import TaggingAgent
from utils import fix_random_state
from utils import training, evaluate
from utils.dict import PieceAlphabet
import os
import sys
import pickle


def run_mpc_autograd_cnn(
    piece_vocab,
    data_house,
    args
):
    """
    Args:
        context_manager: used for setting proxy settings during download.
    """
    crypten.init()
    # # Open the pickle file in read mode
    # with open('/data/home/hemantmishra/examples/CrypTen/data/four_tensor_con.pkl', 'rb') as f:
    #     # Load the pickle data into a Python object
    #     data = pickle.load(f)
    # f.close()
    with open('/data/home/hemantmishra/examples/CrypTen/Co_GAT_1/data/my_file.pkl', 'rb') as f:
        # Load the pickle data into a Python object
        all_data = pickle.load(f)
    f.close()
    input_h, adj, adj_full, adj_re, label_sent, label_act = all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5]
    
    # with open('/data/home/hemantmishra/examples/CrypTen/data/two_label_con.pkl', 'rb') as f:
    #     # Load the pickle data into a Python object
    #     label = pickle.load(f)
    # f.close()
    rank = comm.get().get_rank()
    # assumes at least two parties exist
    # broadcast dummy data with same shape to remaining parties
    
    input_h = torch.cat(input_h, axis=0)
    # input_h.unsqueeze(0)
    adj = torch.cat(adj, axis=0)
    # adj.unsqueeze(0)
    adj_full = torch.cat(adj_full, axis=0)
    # adj_full.unsqueeze(0)
    adj_re = torch.cat(adj_re, axis=0)
    # adj_re.unsqueeze(0)
    label_sent = torch.cat(label_sent, axis=0)
    label_act = torch.cat(label_act, axis=0)

    if rank == 0:
        # a = torch.empty(input_h.size())
        a = input_h
    else:
        a = input_h

    if rank == 0:
        # b = torch.empty(adj.size())
        b = adj
    else:
        b = adj

    if rank == 0:
        c = adj_full
    else:
        # c = torch.empty(adj_full.size())
        c = adj_full

    if rank == 0:
        d = adj_re
    else:
        # d = torch.empty(adj_re.size())
        d = adj_re

    if rank == 0:
        label_1 = label_sent
    else:
        # label_1 = torch.empty(label_sent.size())
        label_1 = label_sent

    if rank == 0:
        label_2 = label_act
    else:
        # label_2 = torch.empty(label_act.size())
        label_2 = label_act

    # encrypt
    # x_reduced = crypten.cryptensor(x_alice, src=0)
    
    a_reduced = crypten.cryptensor(a, src=1)
    b_reduced = crypten.cryptensor(b, src=1)
    c_reduced = crypten.cryptensor(c, src=1)
    d_reduced = crypten.cryptensor(d, src=1)
    y_1_reduced = crypten.cryptensor(label_1, src=0)
    y_2_reduced = crypten.cryptensor(label_2, src=0)
    
    # # combine feature sets
    # x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)
    # x_combined_enc = x_combined_enc.unsqueeze(1)

    # reduce training set to num_samples
    # x_reduced = x_combined_enc[:num_samples]
    # y_reduced = train_labels[:num_samples]

    # encrypt plaintext model
    model = TaggingAgent(
        args.embedding_dim,
        args.hidden_dim, args.num_layer, args.gat_layer, args.gat_dropout_rate, args.dropout_rate,
        args.linear_decoder, args.pretrained_model)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.data_dir == "dataset/mastodon":
        metric = False
    else:
        metric = True
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    onnx_name = "model_onnx_new_0.onnx"
    # dummy_size = (5,34,60)
    # dummy_input = torch.zeros(dummy_size)
    aa = input_h[0].unsqueeze(0)
    bb = adj[0].unsqueeze(0)
    cc = adj_full[0].unsqueeze(0)
    dd = adj_re[0].unsqueeze(0)

    # pytorch_to_onnx(model, input_h[0], tensor_size[0], adj[0], adj_full[0], adj_re[0], onnx_name)
    # torch.onnx.export(model, (aa, bb, dd), onnx_name)
    torch.onnx.export(model, (aa, bb, dd), onnx_name)

    print("ONNX model completed!!")
    # crypten_model = crypten.nn.from_pytorch(model, aa)
    crypten_model = onnx_to_crypten(model, "/data/home/hemantmishra/examples/CrypTen/model_onnx_new_1.onnx")
    # op_type = "LSTM"
    # op_list = [node.op_type for node in crypten_model.graph.node]
    crypten_model.train()
    crypten_model.encrypt()
    print("Training and Encryption of model is completed!!")

    # Mark layers as trainable
    for layer in crypten_model.modules():
        if hasattr(layer, "weight"):
            layer.weight.requires_grad = True
        if hasattr(layer, "bias"):
            layer.bias.requires_grad = True
            
    

    # encrypted training
    train_encrypted(
        a, b_reduced, c_reduced, d_reduced, y_1_reduced, y_2_reduced, crypten_model, args
    )


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
    from onnx import helper, TensorProto
    import copy
    from crypten.nn import onnx_converter
    onnx_model = onnx.load(onnx_model)

    op_type = "LSTM"
    clip_nodes = [node for node in onnx_model.graph.node if node.op_type == op_type]
    for node in clip_nodes:
        node.input[4] = "sequence_lens"
    ## Pop the element
    # clip_nodes[0].input[2] = "max_v"
    Y = helper.make_tensor('sequence_lens', TensorProto.FLOAT, [], [0.])
    onnx_model.graph.initializer.append(Y)

    crypten_model = onnx_converter._to_crypten(onnx_model)
    # crypten_model.pytorch_model = copy.deepcopy(model)
    # make sure training / eval setting is copied:
    crypten_model.train(mode=model.training)
    print("crypten, model training completed")
    return crypten_model

def retrieve(self, tensor, x, y):
    tensor = tensor.unsqueeze(0)
    return tensor
    # return tensor[:,:x,:y]

def train_encrypted(
    input_h,
    adj,
    adj_full,
    adj_re,
    y_1_encrypted,
    y_2_encrypted,
    encrypted_model,
    args
):
    rank = comm.get().get_rank()
    loss = crypten.nn.MSELoss()

    # num_samples = x_encrypted.size(0)
    # label_eye = torch.eye(2)

    for epoch in range(args.num_epoch):
        last_progress_logged = 0
        # only print from rank 0 to avoid duplicates for readability
        if rank == 0:
            print(f"Epoch {epoch} in progress:")
        num_samples_train = input_h.size()[0]
        for j in range(0, num_samples_train, args.batch_size):

            # define the start and end of the training mini-batch
            # start, end = j, min(j + args.batch_size, num_samples_train)

            # switch on autograd for training examples
            # x_train = (input_h[j], adj[j], adj_full[j], adj_re[j])
            # x_train.requires_grad = True
            # t = tensor_size[j]
            # x_train_1 = retrieve(input_h[j], t[0], t[1])
            # x_train_2 = retrieve(adj[j], t[0], t[0])
            # x_train_3 = retrieve(adj_full[j], t[0], t[0])
            # x_train_4 = retrieve(adj_re[j], 2*t[0], 2*t[0])

            x_train_1 = input_h[j]
            x_train_1 = x_train_1.unsqueeze(0)
            # x_train_1.requires_grad = True
            # x_train_size = tensor_size[j]
            # x_train_size.required_grad = True
            x_train_2 = adj[j]
            x_train_2 = x_train_2.unsqueeze(0)
            x_train_2.requires_grad = True
            x_train_3 = adj_full[j]
            x_train_3 = x_train_3.unsqueeze(0)
            x_train_3.requires_grad = True
            x_train_4 = adj_re[j]
            x_train_4 = x_train_4.unsqueeze(0)
            x_train_4.requires_grad = True
            y_train_1 = y_1_encrypted[j]
            y_train_1 = y_train_1.unsqueeze(0)
            y_train_1.requires_grad = True
            y_train_2 = y_2_encrypted[j]
            y_train_2 = y_train_2.unsqueeze(0)
            y_train_2.requires_grad = True
            # y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

            # perform forward pass:
            # output = encrypted_model(x_train_1, x_train_2, x_train_4)
            output = encrypted_model(x_train_1)

            loss_value = loss(output, )

            # backprop
            encrypted_model.zero_grad()
            loss_value.backward()
            encrypted_model.update_parameters(learning_rate)

            # log progress
            if j + batch_size - last_progress_logged >= print_freq:
                last_progress_logged += print_freq
                print(f"Loss {loss_value.get_plain_text().item():.4f}")

        # compute accuracy every epoch
        pred = output.get_plain_text().argmax(1)
        correct = pred.eq(y_encrypted[start:end])
        correct_count = correct.sum(0, keepdim=True).float()
        accuracy = correct_count.mul_(100.0 / output.size(0))

        loss_plaintext = loss_value.get_plain_text().item()
        print(
            f"Epoch {epoch} completed: "
            f"Loss {loss_plaintext:.4f} Accuracy {accuracy.item():.2f}"
        )



def train_encrypted_1(
    x_encrypted,
    y_encrypted,
    model,
    args
):
    dev_best_sent, dev_best_act = 0.0, 0.0
    test_sent_sent, test_sent_act = 0.0, 0.0
    test_act_sent, test_act_act = 0.0, 0.0
    rank = comm.get().get_rank()
    loss = crypten.nn.MSELoss()

    # num_samples = x_encrypted.size(0)
    # label_eye = torch.eye(2)

    for epoch in range(0, args.num_epoch):
        print("Training Epoch: {:4d} ...".format(epoch), file=sys.stderr)

        # xx = data_house.get_iterator("train", args.batch_size, True)
        print(os.getcwd())
        train_loss, train_time = training(model, x_encrypted, y_encrypted, 
                                        10.0, args.bert_learning_rate, args.pretrained_model)
        print("[Epoch{:4d}], train loss is {:.4f}, cost {:.4f} s.".format(epoch, train_loss, train_time))
        
        dev_sent_f1, _, _, dev_act_f1, _, _, dev_time = evaluate(
            model, data_house.get_iterator("dev", args.batch_size, False), metric)
        test_sent_f1, sent_r, sent_p, test_act_f1, act_r, act_p, test_time = evaluate(
            model, data_house.get_iterator("test", args.batch_size, False), metric)

        print("On dev, sentiment f1: {:.4f}, act f1: {:.4f}".format(dev_sent_f1, dev_act_f1))
        print("On test, sentiment f1: {:.4f}, act f1 {:.4f}".format(test_sent_f1, test_act_f1))
        print("Dev and test cost {:.4f} s.\n".format(dev_time + test_time))

        # if get a higher score on dev set, do predict on test set, base on sent or act.
        if dev_sent_f1 > dev_best_sent or dev_act_f1 > dev_best_act:

            if dev_sent_f1 > dev_best_sent:
                dev_best_sent = dev_sent_f1

                test_sent_sent = test_sent_f1
                test_sent_act = test_act_f1

                print("<Epoch {:4d}>, Update (base on sent) test score: sentiment f1: {:.4f} (r: "
                    "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                    ";".format(epoch, test_sent_sent, sent_r, sent_p, test_sent_act, act_r, act_p))

            if dev_act_f1 > dev_best_act:
                dev_best_act = dev_act_f1

                test_act_sent = test_sent_f1
                test_act_act = test_act_f1

                print("<Epoch {:4d}>, Update (base on act) test score: sentiment f1: {:.4f} (r: "
                    "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                    ";".format(epoch, test_act_sent, sent_r, sent_p, test_act_act, act_r, act_p))

            torch.save(model, os.path.join(args.save_dir, "model.pt"))

            print("", end="\n")

def preprocess_mnist(context_manager):
    if context_manager is None:
        context_manager = NoopContextManager()

    with context_manager:
        # each party gets a unique temp directory
        with tempfile.TemporaryDirectory() as data_dir:
            mnist_train = datasets.MNIST(data_dir, download=True, train=True)
            mnist_test = datasets.MNIST(data_dir, download=True, train=False)

    # modify labels so all non-zero digits have class label 1
    mnist_train.targets[mnist_train.targets != 0] = 1
    mnist_test.targets[mnist_test.targets != 0] = 1
    mnist_train.targets[mnist_train.targets == 0] = 0
    mnist_test.targets[mnist_test.targets == 0] = 0

    # compute normalization factors
    data_all = torch.cat([mnist_train.data, mnist_test.data]).float()
    data_mean, data_std = data_all.mean(), data_all.std()
    tensor_mean, tensor_std = data_mean.unsqueeze(0), data_std.unsqueeze(0)

    # normalize data
    data_train_norm = transforms.functional.normalize(
        mnist_train.data.float(), tensor_mean, tensor_std
    )

    # partition features between Alice and Bob
    data_alice = data_train_norm[:, :, :20]
    data_bob = data_train_norm[:, :, 20:]
    train_labels = mnist_train.targets

    return data_alice, data_bob, train_labels


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
