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


# surpress warning of math.floor() returning a float. In Python 3 re