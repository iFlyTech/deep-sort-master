"""
For training models.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn

from utils impo