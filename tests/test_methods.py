#!/usr/bin/python3 python

"""Test methods

"""

import miditok
from torch import Tensor as ptTensor, IntTensor as ptIntTensor, FloatTensor as ptFloatTensor
from tensorflow import Tensor as tfTensor, convert_to_tensor


def test_convert_tensors():
    original = [[2, 6, 87, 89, 25, 15]]
    types = [ptTensor, ptIntTensor, ptFloatTensor, tfTensor]

    def nothing(tokens):
        return tokens

    tokenizer = miditok.TSD()
    for type_ in types:
        if type_ == tfTensor:
            tensor = convert_to_tensor(original)
        else:
            tensor = type_(original)
        tokenizer(tensor)  # to make sure it passes
        as_list = miditok.midi_tokenizer_base.convert_tokens_tensors_to_list(nothing)(tensor)
        assert as_list == original


if __name__ == "__main__":
    test_convert_tensors()
