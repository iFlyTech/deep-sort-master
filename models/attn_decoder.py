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
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(num_embeddings=self.output_dim,
                                      embedding_dim=embedding_dim)
        self.attn = nn.Linear(in_features=self.hidden_dim + embedding_dim,
                              out_features=self.max_length)
        self.attn_combine = nn.Linear(in_features=self.hidden_dim + embedding_dim,
                                      out_features=self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_