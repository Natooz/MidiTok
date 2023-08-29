"""
Common classes.
"""
from dataclasses import dataclass
from typing import Union, Any, List, Sequence, Dict, Tuple
from copy import deepcopy
from pathlib import Path
import json

from numpy import ndarray

from .constants import (
    PITCH_RANGE,
    BEAT_RES,
    NB_VELOCITIES,
    SPECIAL_TOKENS,
    USE_CHORDS,
    USE_RESTS,
    USE_TEMPOS,
    USE_TIME_SIGNATURE,
    USE_SUSTAIN_PEDALS,
    USE_PITCH_BENDS,
    USE_PROGRAMS,
    BEAT_RES_REST,
    CHORD_MAPS,
    CHORD_TOKENS_WITH_ROOT_NOTE,
    CHORD_UNKNOWN,
    NB_TEMPOS,
    TEMPO_RANGE,
    LOG_TEMPOS,
    DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES,
    TIME_SIGNATURE_RANGE,
    SUSTAIN_PEDAL_DURATION,
    PITCH_BEND_RANGE,
    DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES,
    PROGRAMS,
    ONE_TOKEN_STREAM_FOR_PROGRAMS,
    CURRENT_VERSION_PACKAGE,
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
        r"""Checks if too sequences are equal.
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
    :param pitch_range: (default: (21, 109)) range of MIDI pitches to use. Pitches can take
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
    :param use_chords: will use ``Chord`` tokens, if the tokenizer is compatible.
            A `Chord` token indicates the presence of a chord at a certain time step.
            MidiTok uses a chord detection method based on onset times and duration. This allows
            MidiTok to detect precisely chords without ambiguity, whereas most chord detection
            methods in symbolic music based on chroma features can't. Note that using chords will increase
            the tokenization time, especially if you are working on music with a high "note density". (default: False)
    :param use_rests: will use ``Rest`` tokens, if the tokenizer is compatible.
            `Rest` tokens will be placed whenever a portion of time is silent, i.e. no note is being played.
            This token type is decoded as a *TimeShift* event. You can choose the minimum and maximum rests
            values to represent with the `beat_res_rest` parameter. (default: False)
    :param use_tempos: will use ``Tempo`` tokens, if the tokenizer is compatible.
            ``Tempo`` tokens will specify the current tempo. This allows to train a model to predict tempo changes.
            Tempo values are quantized accordingly to the ``nb_tempos`` and ``tempo_range`` entries in the
            ``additional_tokens`` dictionary (default is 32 tempos from 40 to 250). (default: False)
    :param use_time_signatures: will use ``TimeSignature`` tokens, if the tokenizer is compatible.
            `TimeSignature` tokens will specify the current time signature. Note that :ref:`REMI` and :ref:`REMIPlus`
            adds a `TimeSignature` token at the beginning of each Bar (i.e. after `Bar` tokens), while :ref:`TSD` and
            :ref:`MIDILike` will only represent time signature changes (MIDI messages) as they come. If you want more
            "recalls" of the current time signature within your token sequences, you can preprocess you MIDI file to
            add more TimeSignatureChange objects. (default: False)
    :param use_sustain_pedals: will use `Pedal` tokens to represent the sustain pedal events. In multitrack setting,
            The value of each Pedal token will be equal to the program of the track. (default: False)
    :param use_pitch_bends: will use `PitchBend` tokens. In multitrack setting, a `Program` token will be added before
            each `PitchBend` token. (default: False)
    :param use_programs: will use ``Program`` tokens, if the tokenizer is compatible.
            Used to specify an instrument / MIDI program. The :ref:`Octuple`, :ref:`MMM` and :ref:`MuMIDI` tokenizers
            use natively `Program` tokens, this option is always enabled. :ref:`TSD`, :ref:`REMI`, :ref:`MIDILike`,
            :ref:`Structured` and :ref:`CPWord` will add `Program` tokens before each `Pitch` / `NoteOn` token to
            indicate its associated instrument and will treat all the tracks of a MIDI as a single sequence of tokens.
            :ref:`CPWord`, :ref:`Octuple` and :ref:`MuMIDI` add a `Program` tokens with the stacks of `Pitch`,
            `Velocity` and `Duration` tokens. (default: False)
    :param beat_res_rest: the beat resolution of `Rest` tokens. It follows the same data pattern as the `beat_res`
            argument, however the maximum resolution for rests cannot be higher than the highest "global" resolution
            (`beat_res`). Rests are considered complementary to other time tokens (`TimeShift`, `Bar` and `Position`).
            If in a given situation, `Rest` tokens cannot represent time with the exact precision, other time times will
            complement them. (default: `{(0, 1): 8, (1, 2): 4, (2, 12): 2}`)
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
    :param log_tempos: will use log scaled tempo values instead of linearly scaled. (default: False)
    :param delete_equal_successive_tempo_changes: setting this option True will delete identical successive tempo
            changes when preprocessing a MIDI file after loading it. For examples, if a MIDI has two tempo changes
            for tempo 120 at tick 1000 and the next one is for tempo 121 at tick 1200, during preprocessing the tempo
            values are likely to be downsampled and become identical (120 or 121). If that's the case, the second
            tempo change will be deleted and not tokenized. This parameter doesn't apply for tokenizations that natively
            inject the tempo information at recurrent timings (e.g. Octuple). For others, note that setting it True
            might reduce the number of `Tempo` tokens and in turn the recurrence of this information. Leave it False if
            you want to have recurrent `Tempo` tokens, that you might inject yourself by adding `TempoChange` objects to
            your MIDIs. (default: False)
    :param time_signature_range: range as a dictionary {denom_i: [num_i1, ..., num_in] / (min_num_i, max_num_i)}.
            (default: {8: [3, 12, 6], 4: [5, 6, 3, 2, 1, 4]})
    :param sustain_pedal_duration: by default, the tokenizer will use `PedalOff` tokens to mark the offset times of
            pedals. By setting this parameter True, it will instead use `Duration` tokens to explicitly express their
            durations. If you use this parameter, make sure to configure `beat_res` to cover the durations you expect.
            (default: False)
    :param pitch_bend_range: range of the pitch bend to consider, to be given as a tuple with the form
            `(lowest_value, highest_value, nb_of_values)`. There will be `nb_of_values` tokens equally spaced between
             `lowest_value` and `highest_value`. (default: (-8192, 8191, 32))
    :param delete_equal_successive_time_sig_changes: setting this option True will delete identical successive time
            signature changes when preprocessing a MIDI file after loading it. For examples, if a MIDI has two time
            signature changes for 4/4 at tick 1000 and the next one is also 4/4 at tick 1200, the second time signature
            change will be deleted and not tokenized. This parameter doesn't apply for tokenizations that natively
            inject the time signature information at recurrent timings (e.g. Octuple). For others, note that setting it
            True might reduce the number of `TimeSig` tokens and in turn the recurrence of this information. Leave it
            False if you want to have recurrent `TimeSig` tokens, that you might inject yourself by adding
            `TimeSignatureChange` objects to your MIDIs. (default: False)
    :param programs: sequence of MIDI programs to use. Note that `-1` is used and reserved for drums tracks.
            (default: from -1 to 127 included)
    :param one_token_stream_for_programs: when using programs (`use_programs`), this parameters will make the tokenizer
            treat all the tracks of a MIDI as a single stream of tokens. A `Program` token will prepend each `Pitch`,
            `NoteOn` and `NoteOff` tokens to indicate their associated program / instrument. Note that this parameter is
            always set to True for `MuMIDI` and `MMM`. Setting it to False will make the tokenizer not use `Programs`,
             but will allow to still have `Program` tokens in the vocabulary. (default: True)
    :param **kwargs: additional parameters that will be saved in `config.additional_params`.
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
        use_sustain_pedals: bool = USE_SUSTAIN_PEDALS,
        use_pitch_bends: bool = USE_PITCH_BENDS,
        use_programs: bool = USE_PROGRAMS,
        beat_res_rest: Dict[Tuple[int, int], int] = BEAT_RES_REST,
        chord_maps: Dict[str, Tuple] = CHORD_MAPS,
        chord_tokens_with_root_note: bool = CHORD_TOKENS_WITH_ROOT_NOTE,
        chord_unknown: Tuple[int, int] = CHORD_UNKNOWN,
        nb_tempos: int = NB_TEMPOS,
        tempo_range: Tuple[int, int] = TEMPO_RANGE,
        log_tempos: bool = LOG_TEMPOS,
        delete_equal_successive_tempo_changes: bool = DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES,
        time_signature_range: Dict[
            int, Union[List[int], Tuple[int, int]]
        ] = TIME_SIGNATURE_RANGE,
        sustain_pedal_duration: bool = SUSTAIN_PEDAL_DURATION,
        pitch_bend_range: Tuple[int, int, int] = PITCH_BEND_RANGE,
        delete_equal_successive_time_sig_changes: bool = DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES,
        programs: Sequence[int] = PROGRAMS,
        one_token_stream_for_programs: bool = ONE_TOKEN_STREAM_FOR_PROGRAMS,
        **kwargs,
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
        self.use_sustain_pedals: bool = use_sustain_pedals
        self.use_pitch_bends: bool = use_pitch_bends
        self.use_programs: bool = use_programs

        # Rest params
        self.beat_res_rest: Dict[Tuple[int, int], int] = beat_res_rest

        # Chord params
        self.chord_maps: Dict[str, Tuple] = chord_maps
        # Tokens will look as "Chord_C:maj"
        self.chord_tokens_with_root_note: bool = chord_tokens_with_root_note
        # (3, 6) for chords between 3 and 5 notes
        self.chord_unknown: Tuple[int, int] = chord_unknown

        # Tempo params
        self.nb_tempos: int = nb_tempos  # nb of tempo bins for additional tempo tokens, quantized like velocities
        self.tempo_range: Tuple[int, int] = tempo_range  # (min_tempo, max_tempo)
        self.log_tempos: bool = log_tempos
        self.delete_equal_successive_tempo_changes = (
            delete_equal_successive_tempo_changes
        )

        # Time signature params
        self.time_signature_range: Dict[int, List[int]] = {
            beat_res: list(range(beats[0], beats[1] + 1))
            if isinstance(beats, tuple)
            else beats
            for beat_res, beats in time_signature_range.items()
        }
        self.delete_equal_successive_time_sig_changes = (
            delete_equal_successive_time_sig_changes
        )

        # Sustain pedal params
        self.sustain_pedal_duration = sustain_pedal_duration

        # Pitch bend params
        self.pitch_bend_range = pitch_bend_range

        # Programs
        self.programs: Sequence[int] = programs
        self.one_token_stream_for_programs = one_token_stream_for_programs

        # Additional params
        self.additional_params = kwargs

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any], **kwargs):
        r"""
        Instantiates an ``AdditionalTokensConfig`` from a Python dictionary of parameters.

        :param input_dict: Dictionary that will be used to instantiate the configuration object.
        :param kwargs: Additional parameters from which to initialize the configuration object.
        :returns: ``AdditionalTokensConfig``: The configuration object instantiated from those parameters.
        """
        input_dict.update(**input_dict["additional_params"])
        input_dict.pop("additional_params")
        if "miditok_version" in input_dict:
            input_dict.pop("miditok_version")
        config = cls(**input_dict, **kwargs)
        return config

    def to_dict(self, serialize: bool = False) -> Dict[str, Any]:
        r"""
        Serializes this instance to a Python dictionary.

        :param serialize: will serialize the dictionary before returning it, so it can be saved to a JSON file.
        :return: Dictionary of all the attributes that make up this configuration instance.
        """
        dict_config = deepcopy(self.__dict__)
        if serialize:
            self.__serialize_dict(dict_config)
        return dict_config

    def __serialize_dict(self, dict_: Dict):
        r"""
        Converts numpy arrays to lists recursively within a dictionary.

        :param dict_: dictionary to serialize
        """
        for key in dict_:
            if isinstance(dict_[key], dict):
                self.__serialize_dict(dict_[key])
            elif isinstance(dict_[key], ndarray):
                dict_[key] = dict_[key].tolist()

    def save_to_json(self, out_path: Union[str, Path]):
        r"""
        Saves a tokenizer configuration object to the `out_path` path, so that it can be re-loaded later.

        :param out_path: path to the output configuration JSON file.
        """
        if isinstance(out_path, str):
            out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dict_config = self.to_dict(serialize=True)
        for beat_res_key in ["beat_res", "beat_res_rest"]:
            dict_config[beat_res_key] = {
                f"{k1}_{k2}": v for (k1, k2), v in dict_config[beat_res_key].items()
            }
        dict_config["miditok_version"] = CURRENT_VERSION_PACKAGE

        with open(out_path, "w") as outfile:
            json.dump(dict_config, outfile, indent=4)

    @classmethod
    def load_from_json(cls, config_file_path: Union[str, Path]) -> "TokenizerConfig":
        r"""
        Loads a tokenizer configuration from the `config_path` path.

        :param config_file_path: path to the configuration JSON file to load.
        """
        if isinstance(config_file_path, str):
            config_file_path = Path(config_file_path)

        with open(config_file_path) as param_file:
            dict_config = json.load(param_file)

        for beat_res_key in ["beat_res", "beat_res_rest"]:
            dict_config[beat_res_key] = {
                tuple(map(int, beat_range.split("_"))): res
                for beat_range, res in dict_config[beat_res_key].items()
            }

        dict_config["time_signature_range"] = {
            int(res): beat_range
            for res, beat_range in dict_config["time_signature_range"].items()
        }

        return cls.from_dict(dict_config)

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
