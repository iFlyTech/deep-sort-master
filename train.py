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
    decoder_optim = optim(decoder.parameters(), **optim_params)

    checkpoint = load_checkpoint("ptr" if is_ptr else "vanilla")
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        first_iter = checkpoint["iter"]
        plot_losses = checkpoint["plot_losses"]
        print_loss_total = checkpoint["print_loss_total"]
        plot_loss_total = checkpoint["plot_loss_total"]
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder_optim.load_state_dict(checkpoint["encoder_optim"])
        decoder_optim.load_state_dict(checkpoint["decoder_optim"])
    else:
        start_epoch = 0
        first_iter = 0
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        encoder.apply(weight_init)  # init