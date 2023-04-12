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
        encoder.apply(weight_init)  # initialize weights
        decoder.apply(weight_init)  # initialize weights

    criterion = nn.NLLLoss()

    size, n_iters = len(training_pairs), n_epochs * len(training_pairs)
    current_iter = start_epoch * size + first_iter
    start = time.time()
    for epoch in range(start_epoch, n_epochs):
        np.random.shuffle(training_pairs)
        start_iter = first_iter if epoch == start_epoch else 0
        for i in range(start_iter, size):
            loss = train_step(training_pairs[i], encoder, decoder, encoder_optim, decoder_optim, is_ptr, criterion,
                              teacher_force_ratio, grad_clip)
            print_loss_total += loss
            plot_loss_total += loss
            current_iter += 1

            if current_iter % print_every == 0:
                print_loss_avg, print_loss_total = print_loss_total / print_every, 0
                print('%s (epoch: %d iter: %d %d%%) %.4f' % (time_since(start, current_iter / n_iters),
                                                             epoch, i + 1,
                                                             current_iter / n_iters * 100,
                                                             print_loss_avg))

            if current_iter % plot_every == 0:
                plot_loss_avg, plot_loss_total = plot_loss_total / plot_every, 0
                plot_losses.append(plot_loss_avg)

            if current_iter % save_every == 0:
                if i + 1 < size:
 