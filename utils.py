
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
