"""
Dataset classes and data collators to be used with PyTorch when training a model.

``DatasetTok`` is a general i/o class that loads and tokenize MIDI files and saves them
in memory during its initialization, that can chunk the whole token sequences into
smaller sections with a minimum and maximum size. ``DatasetJsonIO`` loads json tokens
files on the fly when it is iterated during batch creations.
"""

from .collators import DataCollator
from .datasets import (
    DatasetJsonIO,
    DatasetMIDI,
    get_distribution_num_tokens_per_bar,
    get_num_beats_for_token_seq_len,
)

__all__ = [
    "DatasetMIDI",
    "DatasetJsonIO",
    "DataCollator",
    "get_distribution_num_tokens_per_bar",
    "get_num_beats_for_token_seq_len",
]
