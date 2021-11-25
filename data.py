
"""
For processing data.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import torch

from utils import device, EOS_token


def str_to_array(lst):
    """
    Converts string representation of array read in from file to Python array.
    """
    temp = lst[1:-1].split(",")
    return [int(i) for i in temp]


def read_data(name, ewc=False):
    """
    Read in data from file.
    """