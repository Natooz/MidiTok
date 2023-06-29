from dataclasses import dataclass
from typing import Union, Any, List, Sequence, Dict, Tuple
from copy import deepcopy

from .constants import (
    PITCH_RANGE,
    BEAT_RES,
    NB_VELOCITIES,
    SPECIAL_TOKENS,
    USE_CHORDS,
    USE_RESTS,
    USE_TEMPOS,
    USE_TIME_SIGNATURE,
    USE_PROGRAMS,
    REST_RANGE,
    CHORD_MAPS,
    CHORD_TOKENS_WITH_ROOT_NOTE,
    CHORD_UNKNOWN,
    NB_TEMPOS,
    TEMPO_RANGE,
    TIME_SIGNATURE_RANGE,
    PROGRAMS,
)


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


class TokenizerConfig:
    r"""
    MIDI tokenizer base class, containing common methods and attributes for all tokenizers.
    :param pitch_range: (default: range(21, 109)) range of MIDI pitches to use. Pitches can take
            values between 0 and 127 (included).
            The `General MIDI 2 (GM2) specifications <https://www.midi.org/specifications-old/item/general-midi-2>`_
            indicate the **recommended** ranges of pitches per MIDI program (instrument).
            These recommended ranges can also be found in ``miditok.constants`` .
            In all cases, the range from 21 to 108 (included) covers all the recommended values.
            When processing a MIDI, the notes with pitches under or above this range can be discarded.
    :param beat_res: (default: `{(0, 4): 8, (4, 12): 4}`) beat resolutions, as a dictionary in the form:
            ``{(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}``
            The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar, and
            the values are the resolution (in samples per beat) to apply to the ranges, ex 8.
            This allows to use **Duration** / **TimeShift** tokens of different lengths / resolutions.
            Note: for tokenization with **Position** tokens, the total number of possible positions will
            be set at four times the maximum resolution given (``max(beat_res.values)``\).
    :param nb_velocities: (default: 32) number of velocity bins. In the MIDI norm, velocities can take
            up to 128 values (0 to 127). This parameter allows to reduce the number of velocity values.
            The velocities of the MIDIs resolution will be downsampled to ``nb_velocities`` values, equally
            separated between 0 and 127.
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "BOS", "EOS", "MASK"]``\)
    :param use_chords: will use ``Chord`` tokens, if the tokenizer is compatible. (default: False)
    :param use_rests: will use ``Rest`` tokens, if the tokenizer is compatible. (default: False)
    :param use_tempos: will use ``Tempo`` tokens, if the tokenizer is compatible. (default: False)
    :param use_time_signatures: will use ``TimeSignature`` tokens, if the tokenizer is compatible. (default: False)
    :param use_programs: will use ``Program`` tokens, if the tokenizer is compatible. (default: False)
    :param rest_range: range of the rest to use, in beats, as a tuple (beat_division, max_rest_in_beats).
            The beat division divides a beat to determine the minimum rest to represent.
            The minimum rest must be divisible by 2 and lower than the first beat resolution
    :param chord_maps: list of chord maps, to be given as a dictionary where keys are chord qualities
            (e.g. "maj") and values pitch maps as tuples of integers (e.g. (0, 4, 7)).
            You can use ``miditok.constants.CHORD_MAPS`` as an example.
    :param chord_tokens_with_root_note: to specify the root note of each chord in ``Chord`` tokens.
            Tokens will look as "Chord_C:maj". (default: False)
    :param chord_unknown: range of number of notes to represent unknown chords.
            If you want to represent chords that does not match any combination in ``chord_maps``, use this argument.
            Leave ``None`` to not represent unknown chords. (default: None)
    :param nb_tempos: number of tempos "bins" to use. (default: 32)
    :param tempo_range: range of minimum and maximum tempos within which the bins fall. (default: (40, 250))
    :param time_signature_range: TODO complete
    :param programs: sequence of MIDI programs to use. Note that `-1` is used and reserved for drums tracks.
            (default: from -1 to 127 included)
    """

    def __init__(
        self,
        pitch_range: Tuple[int, int] = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        use_chords: bool = USE_CHORDS,
        use_rests: bool = USE_RESTS,
        use_tempos: bool = USE_TEMPOS,
        use_time_signatures: bool = USE_TIME_SIGNATURE,
        use_programs: bool = USE_PROGRAMS,
        rest_range: Sequence = REST_RANGE,
        chord_maps: Dict[str, Tuple] = CHORD_MAPS,
        chord_tokens_with_root_note: bool = CHORD_TOKENS_WITH_ROOT_NOTE,
        chord_unknown: Tuple[int, int] = CHORD_UNKNOWN,
        nb_tempos: int = NB_TEMPOS,
        tempo_range: Tuple[int, int] = TEMPO_RANGE,
        time_signature_range: Tuple[int, int] = TIME_SIGNATURE_RANGE,
        programs: Sequence[int] = PROGRAMS,
    ):
        # Global parameters
        self.pitch_range: Tuple[int, int] = pitch_range
        self.beat_res: Dict[Tuple[int, int], int] = beat_res
        self.nb_velocities: int = nb_velocities
        self.special_tokens: Sequence[str] = special_tokens

        # Additional token types params, enabling additional token types
        self.use_chords: bool = use_chords
        self.use_rests: bool = use_rests
        self.use_tempos: bool = use_tempos
        self.use_time_signatures: bool = use_time_signatures
        self.use_programs: bool = use_programs

        # Rest params
        self.rest_range: Sequence = rest_range

        # Chord params
        self.chord_maps: Dict[str, Tuple] = chord_maps
        self.chord_tokens_with_root_note: bool = (
            chord_tokens_with_root_note  # Tokens will look as "Chord_C:maj"
        )
        self.chord_unknown: Tuple[
            int, int
        ] = chord_unknown  # (3, 6) for chords between 3 and 5 notes

        # Tempo params
        self.nb_tempos: int = nb_tempos  # nb of tempo bins for additional tempo tokens, quantized like velocities
        self.tempo_range: Tuple[int, int] = tempo_range  # (min_tempo, max_tempo)

        # Time signature params
        self.time_signature_range: Tuple[int, int] = time_signature_range

        # Programs
        self.programs: Sequence[int] = programs

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any], **kwargs):
        """
        Instantiates an ``AdditionalTokensConfig`` from a Python dictionary of parameters.

        :param input_dict: Dictionary that will be used to instantiate the configuration object.
        :param kwargs: Additional parameters from which to initialize the configuration object.
        :returns: ``AdditionalTokensConfig``: The configuration object instantiated from those parameters.
        """
        config = cls(**input_dict, **kwargs)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        :return: Dictionary of all the attributes that make up this configuration instance.
        """
        return deepcopy(self.__dict__)

    def __eq__(self, other):
        # We don't use the == operator as it yields False when comparing lists and tuples containing the same elements
        # Note: this is not recursive and only checks the first level of iterable values / attributes
        other_dict = other.to_dict()
        for key, value in self.to_dict().items():
            if key not in other_dict:
                return False

            try:
                if len(value) != len(other_dict[key]):
                    return False
                for val1, val2 in zip(value, other_dict[key]):
                    if val1 != val2:
                        return False
            except TypeError:
                if other_dict[key] != value:
                    return False

        return True
