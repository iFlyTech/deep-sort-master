
"""
A decoder in seq2seq model using pointer attention.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device


# ignore unresolved reference for torch.stack, torch.cat (bug in PyTorch)
# ignore that parameter <input> for <forward()> shadows built-in keyword input
# noinspection PyUnresolvedReferences,PyShadowingBuiltins
class PtrDecoder(nn.Module):
    """A decoder in seq2seq model using pointer attention.
    """
    def __init__(self, output_dim,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0.1):
        super(PtrDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings=self.output_dim,
                                      embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=num_layers)
        self.attn = nn.Linear(in_features=self.hidden_dim * 2,
                              out_features=self.hidden_dim)
        self.out = nn.Linear(in_features=self.hidden_dim,
                             out_features=1)

    def forward(self, input, hidden, encoder_outputs, encoder_inputs):
        """
        The forward pass of the decoder.
        """
        length = encoder_outputs.shape[0]  # the length of the input sequence

        # feed input through embedding, dropout, and LSTM cell to get new hidden state
        embedded = self.embedding(input).view(1, 1, -1)