from .midi_tokenizer import MIDITokenizer, convert_sequence_to_tokseq
from .classes import Event, TokSequence, TokenizerConfig
from .tokenizations import (
    MIDILike,
    REMI,
    REMIPlus,
    TSD,
    Structured,
    Octuple,
    OctupleMono,
    CPWord,
    MuMIDI,
    MMM,
)

from .utils import utils
from miditok import data_augmentation


__all__ = [
    "MIDITokenizer",
    "convert_sequence_to_tokseq",
    "Event",
    "TokSequence",
    "TokenizerConfig",
    "MIDILike",
    "REMI",
    "REMIPlus",
    "TSD",
    "Structured",
    "Octuple",
    "OctupleMono",
    "CPWord",
    "MuMIDI",
    "MMM",
    "utils",
    "data_augmentation",
]

try:
    from miditok import pytorch_data
    __all__.append("pytorch_data")
except ImportError as e:
    pass
