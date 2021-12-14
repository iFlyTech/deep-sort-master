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
    Run the experiment.
    """
    is_ptr = True
    np.random.seed(RANDOM_SEED)
    max_val, max_length, pairs = read_data(name="test")
    np.random.shuffle(pairs)
    training_pairs = [tensors_from_pair(pair) for pair in pairs]

    data_dim = max_val + 1
    hidden_dim = embedding_dim = 256

    encoder = Encoder(input_dim=data_dim,
                      embedding_dim=embedding_dim,
                      hidden_dim=hidden_dim).to(device)
    if is_ptr:
        decoder = PtrDecoder(output_dim=data_dim,
            