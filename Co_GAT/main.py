import os
import sys
import json
import torch
import argparse
sys.path.append('/data/home/hemantmishra/CrypTen')
import crypten

from utils import DataHub
from nn import TaggingAgent
from utils import fix_random_state
from utils import training, evaluate
from utils.dict import PieceAlphabet

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
parser = argparse.ArgumentParser()
# Pre-train Hyper parameter
parser.add_argument("--pretrained_model", "-pm", type=str, default="none",
                    choices=["none", "bert", "roberta", "xlnet", "albert", "electra"],
                    help="choose pretrained model, default is none.")
parser.add_argument("--linear_decoder", "-ld", action="store_true", default=False,
                    help="Using Linear decoder to get category.")
parser.add_argument("--bert_learning_rate", "-blr", type=float, default=1e-5,
                    help="The learning rate of all types of pretrain model.")
# Basic Hyper parameter
parser.add_argument("--data_dir", "-dd", type=str, default="Co_GAT/dataset/mastodon")
parser.add_argument("--save_dir", "-sd", type=str, default="./save")
parser.add_argument("--batch_size", "-bs", type=int, default=1)
parser.add_argument("--num_epoch", "-ne", type=int, default=1)
parser.add_argument("--random_state", "-rs", type=int, default=0)

# Model Hyper parameter
parser.add_argument("--num_layer", "-nl", type=int, default=2,
                    help="This parameter CAN NOT be modified! Please use gat_layer to set the layer num of gat")
parser.add_argument("--gat_layer", "-gl", type=int, default=2,
                    help="Control the number of GAT layers. Must be between 2 and 4.")
parser.add_argument("--embedding_dim", "-ed", type=int, default=128)
parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
parser.add_argument("--gat_dropout_rate", "-gdr", type=float, default=0.1)

args = parser.parse_args()
print(json.dumps(args.__dict__, indent=True), end="\n\n\n")

# fix random seed
fix_random_state(args.random_state)

# Build dataset
data_house = DataHub.from_dir_addadj(args.data_dir)
# piece vocab
piece_vocab = PieceAlphabet("piece", pretrained_model=args.pretrained_model)

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

# #Setting model variable in model_variable file
import model_variable
obj = model_variable.model_variable()
obj.set_model(model)
# xx = obj.get_model()

# Exporting model to onnx name="model_16.onnx"
def pytorch_to_onnx(model, dummy_input, onnx_name):
    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_name)
    print("ONNX model completed!!")

# Going to train on crypten model
def onnx_to_crypten(model, onnx_model):
    import sys
    sys.path.append('/data/home/hemantmishra/CrypTen/')
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

import os

print(os.environ.get("WORLD_SIZE", 1))
print(os.environ.get("RANK", 0))

# crypten.init()

# input_h = torch.empty((1,17,60))
# len_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# adj = torch.empty((1,17,17))
# adj_full = torch.empty((1,17,17))
# adj_re = torch.empty((1,34,34))

# if torch.cuda.is_available():
#     input_h = input_h.cuda()
#     adj = adj.cuda()
#     adj_full = adj_full.cuda()
#     adj_re = adj_re.cuda()

# onnx_name = "model_1_17_60.onnx"
# dummy_input = (input_h, len_list, adj, adj_full, adj_re)
# pytorch_to_onnx(model, dummy_input, onnx_name)

# crypten_model = onnx_to_crypten(model, onnx_name)
# crypten_model.train()
# crypten_model.encrypt()

# loss = crypten.nn.MSELoss()

# train_encrypted(crypten_model)

# crypten_model = onnx_to_crypten(model, "/data/home/hemantmishra/examples/CrypTen/Co_GAT/model_1_17_60.onnx")
# crypten_model.train()
# crypten_model.encrypt()
# print("Training and Encryption of model is completed!!")


dev_best_sent, dev_best_act = 0.0, 0.0
test_sent_sent, test_sent_act = 0.0, 0.0
test_act_sent, test_act_act = 0.0, 0.0
var_utt, len_list, var_adj, var_adj_full, var_adj_R = [], [], [], [], []
len_train, len_test, len_val = [], [], []
data_train = (var_utt, len_list, var_adj, var_adj_full, var_adj_R, len_train)
label_train, label_val, label_test = [], [], []
data_val = (var_utt, len_list, var_adj, var_adj_full, var_adj_R, len_test)
data_test = (var_utt, len_list, var_adj, var_adj_full, var_adj_R, len_val)
# len_train = ([])
# len_test = ([])
# len_val = ([])


for epoch in range(0, args.num_epoch):
    print("Training Epoch: {:4d} ...".format(epoch), file=sys.stderr)

    xx = data_house.get_iterator("train", args.batch_size, True)
    
    train_loss, train_time = training(model, data_house.get_iterator("train", args.batch_size, True),
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

# a = []
# for list in data_train[1]:
#     for x in list:
#         a.append(len(x))
# print(a)
# with open("data/train_data_my.txt", "w") as f:
#     # Write each list as a separate line, separating elements with a comma
#     for lst in data_train:
#         f.write(",".join(str(x) for x in lst) + "\n")
# f.close()
# tensor_lists = []
# for tensor_list in data_train:
#     if isinstance(tensor_list, torch.Tensor):
#         tensor_list = [x.tolist() for x in tensor.tolist() for tensor in tensor_list]
#     tensor_lists.append(tensor_list)

# # create a dictionary with the tensor lists
# data_dict = {'tensor_lists': tensor_lists}

# # serialize the data to JSON
# json_str = json.dumps(data_dict)

# # write the JSON data to a file
# with open('data.json', 'w') as f:
#     f.write(json_str)
# f.close()