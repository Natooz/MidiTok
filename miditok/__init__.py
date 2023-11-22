from miditok import data_augmentation

from .classes import Event, TokenizerConfig, TokSequence
from .midi_tokenizer import MIDITokenizer, convert_sequence_to_tokseq
from .tokenizations import (
    MMM,
    REMI,
    TSD,
    CPWord,
    MIDILike,
    MuMIDI,
    Octuple,
    Structured,
)
from .utils import utils


class REMIPlus(REMI):
    r"""REMI+ is an extended version of :ref:`REMI` (Huang and Yang) for general
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`, which handle multiple instruments by
    adding `Program` tokens before the `Pitch` ones.

    This class is identical to :ref:`REMI` with `Program` and `TimeSignature` tokens enabled.
    """

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()
        self.config.use_programs = True
        self.config.use_time_signatures = True
        self.one_token_stream = True


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
    "CPWord",
    "MuMIDI",
    "MMM",
    "utils",
    "data_augmentation",
]

try:
    from miditok import pytorch_data  # noqa: F401

    __all__.append("pytorch_data")
except ImportError:
    pass
