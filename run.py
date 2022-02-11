

import numpy as np
import tqdm
import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from utils import read_timeseries,generate_sequence, plt_lmbda
from module import GTPP

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='exponential_hawkes')
    # parser.add_argument("--model", type=str, default='GTPP')
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--event_class", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prt_evry", type=int, default=15)
    # parser.add_argument("--early_stop", type=bool, default=True) # on by default
    ## Alpha ??
    parser.add_argument("--alpha", type=float, default=0.05, help='future discount factor for display true event probability')

    # parser.add_argument("--importance_weight", action="store_true") # not used
    parser.add_argument("--log_mode", type=bool, default=False, help="generate sequence in log mode")

    parser.add_argument("--log_t", action="store_true", help="use log of time in model inputs")
    parser.add_argument("--mean_first", action="store_true", help="in model take mean first")
    return parser













