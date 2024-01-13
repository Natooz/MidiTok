"""
Data augmentation module.

The module implements three public methods:
* :py:func:`miditok.data_augmentation.augment_midi`: augment a unique midi on a unique
set of offsets;
* :py:func:`miditok.data_augmentation.augment_midi_multiple_offsets`: augment a unique
MIDI on combinations of offsets;
* :py:func:`miditok.data_augmentation.augment_midi_dataset`: augment a list of MIDI
files on combinations of offsets.

"""

from .data_augmentation import (
    augment_midi,
    augment_midi_dataset,
    augment_midi_multiple_offsets,
)

__all__ = [
    "augment_midi",
    "augment_midi_multiple_offsets",
    "augment_midi_dataset",
]
