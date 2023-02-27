from dataclasses import dataclass
from typing import Union, Any, List


@dataclass
class Event:
    r"""Event class, representing a token and its characteristics
    The type corresponds to the token type (e.g. Pitch, Position ...);
    The value to its value.
    These two attributes are used to build its string representation (__str__),
    used in the Vocabulary class to map an event to its corresponding token.
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
class Sequence:
    r"""
    """
    tokens: List[str] = None
    ids: List[int] = None  # BPE can be applied on ids
    bytes: str = None
    events: List[Event] = None

    def __len__(self):
        return len(self.ids)
