"""Vocabulary class

"""

from typing import List, Tuple, Union, Generator


class Event:
    r"""Event class, representing a token and its characteristics
    The type corresponds to the token type (e.g. Pitch, Position ...);
    The value to its value.
    These two attributes are used to build its string representation (__str__),
    used in the Vocabulary class to map an event to its corresponding token.
    """

    def __init__(
        self, type_: str, value: Union[str, int], time: Union[int, float] = 0, desc=None
    ):
        self.type = type_
        self.value = value
        self.time = time
        self.desc = desc

    def __str__(self):
        return f"{self.type}_{self.value}"

    def __repr__(self):
        return f"Event(type={self.type}, value={self.value}, time={self.time}, desc={self.desc})"


class Vocabulary:
    r"""Vocabulary class.
    Get an element of the vocabulary from its index, such as:
        token = vocab['Pitch_80']  # gets the token of this event
        event = vocab[140]  # gets the event corresponding to token 140
    You can also use the event_to_token and token_to_event properties,
    which will be faster if you run this in heavy loops.

    Use add_event or the += operator to add an event to the vocab.
    Read add_event docstring for how to give arguments.

    :param pad: will include a PAD token, used when training a model with batch of sequences of
                unequal lengths, and usually at index 0 of the vocabulary.
                If this argument is set to True, the PAD token will be at index 0. (default: True)
    :param sos_eos: will include Start Of Sequence (SOS) and End Of Sequence (tokens) (default: False)
    :param mask: will add a MASK token to the vocabulary. (default: False)
    :param events: a list of events to add to the vocabulary when creating it. (default: None)
    """

    def __init__(
        self,
        pad: bool = True,
        mask: bool = False,
        sos_eos: bool = False,
        events: List[Union[str, Event]] = None,
    ):
        self._event_to_token = {}
        self._token_to_event = {}
        self._token_types_indexes = {}

        # Adds (if specified) special tokens first
        if pad:
            self.__add_pad()
        if sos_eos:
            self.__add_sos_eos()
        if mask:
            self.__add_mask()

        # Add custom events and updates _token_types_indexes
        if events is not None:
            self.add_event(event for event in events)

    def add_event(self, event: Union[Event, str, Generator]):
        r"""Adds one or multiple entries to the vocabulary.

        :param event: event to add, either as an Event object or string of the form "Type_Value", e.g. Pitch_80
        """
        if isinstance(event, Generator):
            while True:
                try:
                    self.__add_distinct_event(str(next(event)))
                except StopIteration:
                    return
        else:
            self.__add_distinct_event(str(event))

    def __add_distinct_event(self, event: Union[str, Event]):
        r"""Private: Adds an event to the vocabulary. Its index (int) will be the length of the vocab.

        :param event: event to add, as a formatted string of the form "Type_Value", e.g. Pitch_80
        """
        if isinstance(event, str):
            event_str = event
            event_type = event.split("_")[0]
        else:
            event_str = str(event)
            event_type = event.type

        index = len(self._token_to_event)
        self._event_to_token[event_str] = index
        self._token_to_event[index] = event_str

        if event_type in self._token_types_indexes:
            self._token_types_indexes[event_type].append(index)
        else:
            self._token_types_indexes[event_type] = [index]

    def token_type(self, token: int) -> str:
        r"""Returns the type of the given token.

        :param token: token to get type from
        :return: the type of the token, as a string
        """
        return self._token_to_event[token].split("_")[0]

    def update_token_types_indexes(self):
        r"""Updates the _token_types_indexes attribute according to _event_to_token."""
        for event, token in self._event_to_token.items():
            token_type = event.split("_")[0]
            if token_type in self._token_types_indexes:
                self._token_types_indexes[token_type].append(token)
            else:
                self._token_types_indexes[token_type] = [token]

    def tokens_of_type(self, token_type: str) -> List[int]:
        r"""Returns the list of tokens of the given type.

        :param token_type: token type to get the associated tokens
        :return: list of tokens
        """
        try:
            return self._token_types_indexes[token_type]
        except KeyError:  # no tokens of this type, returns an empty list
            return []

    def __add_pad(self):
        r"""Adds a PAD token to the vocabulary. It is usually at index 0."""
        self.__add_distinct_event("PAD_None")

    def __add_sos_eos(self):
        r"""Adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens
        to the vocabulary.
        """
        self.__add_distinct_event("SOS_None")
        self.__add_distinct_event("EOS_None")

    def __add_mask(self):
        r"""Adds a MASK token to the vocabulary. This may be used to
        pre-train a model, such as for BERT, before finetuning it.
        """
        self.__add_distinct_event("MASK_None")

    def __getitem__(self, item: Union[int, str]) -> Union[str, int]:
        if isinstance(item, str):
            return self._event_to_token[item]
        elif isinstance(item, int):
            return self._token_to_event[item]
        else:
            raise IndexError("The index must be an integer or a string")

    def __len__(self) -> int:
        return len(self._event_to_token)

    def __iadd__(
        self, other: Union[Generator, Event, str, Tuple[Union[str, Event], int]]
    ):
        self.add_event(*other if isinstance(other, tuple) else other)
        return self

    def __eq__(self, other) -> bool:
        """Checks if another vocabulary is identical.

        :param other: vocabulary to inspect
        :return: True if _event_to_token, _token_to_event and _token_types_indexes attributes are identical
                for both Vocabularies, False otherwise.
        """
        if isinstance(other, Vocabulary):
            if all(
                [
                    self._event_to_token == other._event_to_token,
                    self._token_to_event == other._token_to_event,
                    self._token_types_indexes == other._token_types_indexes,
                ]
            ):
                return True
            return False
        return False

    def __repr__(self):
        return f"Vocabulary - {len(self._event_to_token)} tokens of {len(self._token_types_indexes)} types"

    @property
    def event_to_token(self):
        return self._event_to_token

    @property
    def token_to_event(self):
        return self._token_to_event

    @property
    def token_types(self) -> List[str]:
        return list(self._token_types_indexes.keys())
