
"""
Training a model using the EWC method.

Modified from https://github.com/moskomule/ewc.pytorch
"""
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import SOS_token, device, EOS_token, time_since, save_checkpoint, load_checkpoint


def train(encoder, decoder, optim, optim_params, importance, weight_init, grad_clip, is_ptr, tasks, n_epochs,
          teacher_force_ratio, print_every, plot_every, save_every):
    """
    Train on an assortment of tasks.
    """
    encoder_optim = optim(encoder.parameters(), **optim_params)
    decoder_optim = optim(decoder.parameters(), **optim_params)

    checkpoint = load_checkpoint("ewc_ptr" if is_ptr else "ewc_vanilla")
    if checkpoint:
        start_task = checkpoint["task"]
        first_epoch = checkpoint["epoch"]
        first_iter = checkpoint["iter"]
        plot_losses = checkpoint["plot_losses"]
        print_loss_total = checkpoint["print_loss_total"]
        plot_loss_total = checkpoint["plot_loss_total"]
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder_optim.load_state_dict(checkpoint["encoder_optim"])
        decoder_optim.load_state_dict(checkpoint["decoder_optim"])
    else:
        start_task = 0
        first_epoch = 0
        first_iter = 0
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        encoder.apply(weight_init)  # initialize weights
        decoder.apply(weight_init)  # initialize weights

    current_iter = sum([len(tasks[i]) for i in range(start_task)]) * n_epochs + len(tasks[start_task]) * first_epoch + \
        first_iter
    start = time.time()
    for task in range(start_task, len(tasks)):
        training_pairs = deepcopy(tasks[task])
        size, n_iters = len(training_pairs), n_epochs * len(training_pairs)
        start_epoch = first_epoch if task == start_task else 0
        for epoch in range(start_epoch, n_epochs):
            np.random.shuffle(training_pairs)
            start_iter = first_iter if epoch == start_epoch else 0
            for i in range(start_iter, size):
                loss = train_step(training_pairs[i], tasks[:task], encoder, decoder, encoder_optim, decoder_optim,
                                  is_ptr, F.cross_entropy, importance, teacher_force_ratio, grad_clip)
                print_loss_total += loss
                plot_loss_total += loss
                current_iter += 1

                if current_iter % print_every == 0: