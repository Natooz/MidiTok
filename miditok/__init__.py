from .midi_tokenizer import MIDITokenizer, convert_sequence_to_tokseq
from .classes import Event, TokSequence, TokenizerConfig
from .tokenizations import (
    MIDILike,
    REMI,
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


class REMIPlus(REMI):
    r"""REMI+ is an extended version of :ref:`REMI` (Huang and Yang) for general
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`, which handle multiple instruments by
    adding `Program` tokens before the `Pitch` ones.

    This class is identical to :ref:`REMI` with `Program` and `TimeSignature` tokens enabled.
    """

    def _tweak_config_before_creating_voc(self):
        self.config.use_programs = True
        self.config.use_time_signatures = True
        self.one_token_stream = True

        # In case the tokenizer has been created without specifying any config or params file path
        if "max_bar_embedding" not in self.config.additional_params:
            # If used, this attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
            self.config.additional_params["max_bar_embedding"] = None


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
