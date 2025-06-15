"""
Root module.

Here we only import tokenizer classes and submodules.
"""

from miditok import data_augmentation

from .classes import Event, TokenizerConfig, TokSequence
from .midi_tokenizer import MusicTokenizer
from .tokenizations import (
    MMM,
    REMI,
    TSD,
    CPWord,
    MIDILike,
    MuMIDI,
    Octuple,
    PerTok,
    Structured,
)
from .tokenizer_training_iterator import TokTrainingIterator

__all__ = [
    "MMM",
    "REMI",
    "TSD",
    "CPWord",
    "Event",
    "MIDILike",
    "MuMIDI",
    "MusicTokenizer",
    "Octuple",
    "PerTok",
    "Structured",
    "TokSequence",
    "TokTrainingIterator",
    "TokenizerConfig",
    "data_augmentation",
    "utils",
]

try:
    from miditok import pytorch_data  # noqa: F401

    __all__.append("pytorch_data")
except ImportError:
    pass
