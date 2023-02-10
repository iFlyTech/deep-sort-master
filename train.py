"""
For training models.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn

from utils import device, SOS_token, EOS_token, time_since, save_checkpoint, load_checkpoint, RANDOM_SEED
from visual import show_plot


# surpress warning of math.floor() returning a float. In Python 3 returns it returns an int.
# noinspection PyTypeChecker
def train(encoder, decoder, optim, optim_params, weight_init, grad_clip, is_ptr, training_pairs, n_epochs,
          teacher_force_ratio, print_every, plot_every, save_every):
    """
    The training loop.
    """
    np.random.seed(RANDOM_SEED), torch.manual_seed(RANDOM_SEED)
    encoder.train(), decoder.train()
    encoder_optim = optim(encoder.parameters(), **optim_params)
    decoder_optim = optim(decoder.parame