"""
Simple experiment testing EWC training procedure.
"""
import torch.nn as nn
from torch import optim

from data import read_data, tensors_from_pair
from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder
from train_ewc import train
from utils import device, set_max_length


def weight_init(module):
    """
    Initialize weights of <module>. Applied recursivly over model weights via .apply()
    """
    for parameter in module.parameters():
        nn.init.uniform_(parameter, -0.08, 0.08)


def run():
    """
    Run the experiment.
    """
    max_val, max_length, tasks = read_data(name="ewc",
    