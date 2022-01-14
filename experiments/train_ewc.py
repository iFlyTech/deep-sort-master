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


de