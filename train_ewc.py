
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