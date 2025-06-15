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
    augment_dataset,
    augment_score,
    augment_score_multiple_offsets,
)

__all__ = [
    "augment_dataset",
    "augment_score",
    "augment_score_multiple_offsets",
]
