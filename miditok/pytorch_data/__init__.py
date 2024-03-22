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
)
from .split_midi_utils import (
    get_average_num_tokens_per_note,
    get_distribution_num_tokens_per_beat,
    get_num_beats_for_token_seq_len,
    split_dataset_to_subsequences,
    split_midi_per_note_density,
    split_midis_for_training,
    split_seq_in_subsequences,
)

__all__ = [
    "DatasetMIDI",
    "DatasetJsonIO",
    "DataCollator",
    "get_average_num_tokens_per_note",
    "get_distribution_num_tokens_per_beat",
    "get_num_beats_for_token_seq_len",
    "split_midis_for_training",
    "split_midi_per_note_density",
    "split_dataset_to_subsequences",
    "split_seq_in_subsequences",
]
