from dataclasses import dataclass
from typing import Union, Any, List


@dataclass
class Event:
    r"""Event class, representing a token and its characteristics
    The type corresponds to the token type (e.g. Pitch, Position ...) and its value.
    These two attributes are used to build its string representation (__str__),
    used in the Vocabulary class to map an event to its corresponding token.
    This class is mainly used during tokenization when the tokens / events have
    to be sorted by time.
    """
    type: str
    value: Union[str, int]
    time: Union[int, float] = None
    desc: Any = None

    def __str__(self):
        return f"{self.type}_{self.value}"

    def __repr__(self):
        return f"Event(type={self.type}, value={self.value}, time={self.time}, desc={self.desc})"


@dataclass
class TokSequence:
    r"""Represents a sequence of token.
    The TokSequence class can represent tokens by their several forms:
    * tokens (list of str): tokens as sequence of strings.
    * ids (list of int), these are the one to be fed to models.
    * events (list of Event): Event objects that can carry time or other information useful for debugging.
    * bytes (str): ids are converted into unique bytes, all joined together in a single string. This
        form can be used internally for Byte Pair Encoding.

    The ``ids_are_bpe_encoded`` attribute tells if ``ids`` is encoded with BPE.

    :py:meth:`miditok.MIDITokenizer.complete_sequence`
    """
    tokens: List[Union[str, List[str]]] = None
    ids: List[Union[int, List[int]]] = None  # BPE can be applied on ids
    bytes: str = None
    events: List[Union[Event, List[Event]]] = None
    ids_bpe_encoded: bool = False
    _ids_no_bpe: List[Union[int, List[int]]] = None

    def __len__(self):
        return len(self.ids)
