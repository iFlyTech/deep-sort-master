
"""
Miscellaneous utility functions.
as_minutes(), time_since() modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
save_checkpoint(), load_checkpoint() modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import math
import os
import time
import re

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 16

RANDOM_SEED = 0


def set_max_length(max_length):
    """
    Set the max length of an input sequence.
    """
    global MAX_LENGTH
    MAX_LENGTH = max_length


def as_minutes(s):
    """
    Returns <s> seconds in (hours, minutes, seconds) format.
    """
    h, m = math.floor(s / 3600), math.floor(s / 60)
    m, s = m - h * 60, s - m * 60
    return '%dh %dm %ds' % (h, m, s)


def time_since(since, percent):
    """
    Return time since.
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))