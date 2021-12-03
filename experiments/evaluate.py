"""
Evaluate the model.
"""
import numpy as np

from data import read_data, tensors_from_pair
from evaluate import evaluate
from models.attn_decoder import AttnDecoder
from models.encoder import Encoder
from models.ptr_decoder import PtrDecoder
from utils import load_checkpoint, device, RANDOM_SEED


def run():
    """
    Run th