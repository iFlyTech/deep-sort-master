"""
A simple encoder in the seq2seq model using a LSTM.
"""
import torch
import torch.nn as nn

from utils import device


# ignore that parameter <input> for <forward()> shadows built-in keyword input
# noinspection PyShadowingBuiltins
class Encoder(nn.Module):
    """
    A simple encoder in the seq2seq model using a LSTM.
    """
    def __init__(self, input_dim,
                 embeddi