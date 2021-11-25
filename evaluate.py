
"""
For performing inference using the model.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

import torch

from utils import device, SOS_token, EOS_token, MAX_LENGTH, RANDOM_SEED


# noinspection PyCallingNonCallable,PyUnresolvedReferences
def evaluate(encoder, decoder, input_tensor, is_ptr, max_length=MAX_LENGTH):
    """
    Perform inference using the model.
    """
    torch.manual_seed(RANDOM_SEED)
    encoder.eval(), decoder.eval()
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()
