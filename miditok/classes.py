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
    A ``TokSequence`` can represent tokens by their several forms:

    * tokens (list of str): tokens as sequence of strings.
    * ids (list of int), these are the one to be fed to models.
    * events (list of Event): Event objects that can carry time or other information useful for debugging.
    * bytes (str): ids are converted into unique bytes, all joined together in a single string.

    Bytes are used internally by MidiTok for Byte Pair Encoding.
    The ``ids_are_bpe_encoded`` attribute tells if ``ids`` is encoded with BPE.

    :py:meth:`miditok.MIDITokenizer.complete_sequence`
    """
    tokens: List[Union[str, List[str]]] = None
    ids: List[Union[int, List[int]]] = None  # BPE can be applied on ids
    bytes: str = None
    events: List[Union[Event, List[Event]]] = None
    ids_bpe_encoded: bool = False
    _ids_no_bpe: List[Union[int, List[int]]] = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, item):
        if self.ids is not None:
            return self.ids[item]
        elif self.tokens is not None:
            return self.tokens[item]
        elif self.events is not None:
            return self.events[item]
        elif self.bytes is not None:
            return self.bytes[item]
        elif self._ids_no_bpe is not None:
            return self._ids_no_bpe[item]
        else:
            return ValueError(
                "This TokSequence seems to not be initialized, all its attributes are None."
            )

    def __eq__(self, other) -> bool:
        """Checks if too sequences are equal.
        This is performed by comparing their attributes (ids, tokens...).
        **Both sequences must have at least one common attribute initialized (not None) for this method to work,
        otherwise it will return False.**

        :param other: other sequence to compare.
        :return: True if the sequences have equal attributes.
        """
        # Start from True assumption as some attributes might be unfilled (None)
        attributes = ["tokens", "ids", "bytes", "events"]
        eq = [True for _ in attributes]
        common_attr = False
        for i, attr in enumerate(attributes):
            if getattr(self, attr) is not None and getattr(other, attr) is not None:
                eq[i] = getattr(self, attr) == getattr(other, attr)
                common_attr = True

        return all(eq) if common_attr else False
