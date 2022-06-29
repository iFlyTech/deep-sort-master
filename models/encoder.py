"""
A simple encoder in the seq2seq model using a LSTM.
"""
import torch
import torch.nn as nn

from utils import device


# ignore that parameter <input> for <forward()> shadows built-in keyword input
# noinspection PyShadowingBuiltins
class Encoder(nn.Mo