"""
A simple encoder in the seq2seq model using a LSTM.
"""
import torch
import torch.nn as nn

from utils import device


# ignore that parameter <input> for <forwa