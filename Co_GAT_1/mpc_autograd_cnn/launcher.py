#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
To run mpc_autograd_cnn example:

$ python examples/mpc_autograd_cnn/launcher.py

To run mpc_linear_svm example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_autograd_cnn/mpc_autograd_cnn.py \
      examples/mpc_autograd_cnn/launcher.py
"""

import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import sys
sys.path.append('/data/home/hemantmishra/examples/CrypTen')

from Co_GAT_1.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="CrypTen Autograd Co-GAT Training")


def validate_world_size(world_size):
    world_size = int(world_size)
    if world_size < 2:
        raise argparse.ArgumentTypeError(f"world_size {world_size} must be > 1")
    return world_size

import json
import torch
import argparse
sys.path.append('/data/home/hemantmishra/examples/CrypTen/Co_GAT_1')
import crypten

from utils import DataHub

from nn import TaggingAgent
from utils import fix_random_state
from utils import training, evaluate
from utils.dict import PieceAlphabet


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
parser.add_argument("--data_dir", "-dd", type=str, default="/data/home/hemantmishra/examples/CrypTen/Co_GAT_1/dataset/mastodon_updated")
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

parser.add_argument(
    "--world_size",
    type=validate_world_size,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

args = parser.parse_args()
print(json.dumps(args.__dict__, indent=True), end="\n\n\n")

# fix random seed
fix_random_state(args.random_state)

# Build dataset
# data_house = DataHub.from_dir_addadj(args.data_dir)
# # piece vocab
# piece_vocab = PieceAlphabet("piece", pretrained_model=args.pretrained_model)


def _run_experiment(args):
    level = logging.INFO
    # Build dataset
    data_house = DataHub.from_dir_addadj(args.data_dir)
    # piece vocab
    piece_vocab = PieceAlphabet("piece", pretrained_model=args.pretrained_model)
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from mpc_autograd_cnn import run_mpc_autograd_cnn

    run_mpc_autograd_cnn(
        piece_vocab,
        data_house,
        args
    )


def main(run_experiment):
    args = parser.parse_args()
    # run multiprocess by default
    launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
    launcher.start()
    launcher.join()
    launcher.terminate()


if __name__ == "__main__":
    main(_run_experiment)
