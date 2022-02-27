
"""
This module generates the data for the sorting task.
"""
import os

import numpy as np

from config.generate import get_config
from utils import RANDOM_SEED


def generate(name, size, max_val, min_length, max_length, ewc):
    """Generates <size> samples for the dataset,
    """
    np.random.seed(RANDOM_SEED)
    if not os.path.isdir("data"):
        os.mkdir("data")
    with open("data/" + name + ".txt", mode='w') as file:
        file.write("|".join([str(size), str(max_val), str(max_length)]) + "\n")