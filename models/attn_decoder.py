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
    """

    def __init__(self, output_dim,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.