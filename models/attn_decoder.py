"""
The attention decoder model.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device, MAX_LENGTH


# noinspection PyUnresolvedReferences,PyShadowingBuiltins
class AttnDecoder(nn.Module):
    """A decoder in seq2seq model using a gated recurrent unit (GRU) and attention.
   