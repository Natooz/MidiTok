"""Common classes."""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy import ndarray

from .constants import (
    BEAT_RES,
    BEAT_RES_REST,
    CHORD_MAPS,
    CHORD_TOKENS_WITH_ROOT_NOTE,
    CHORD_UNKNOWN,
    CURRENT_MIDITOK_VERSION,
    CURRENT_SYMUSIC_VERSION,
    CURRENT_TOKENIZERS_VERSION,
    DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES,
    DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES,
    DRUM_PITCH_RANGE,
    LOG_TEMPOS,
    MAX_PITCH_INTERVAL,
    NUM_TEMPOS,
    NUM_VELOCITIES,
    ONE_TOKEN_STREAM_FOR_PROGRAMS,
    PITCH_BEND_RANGE,
    PITCH_INTERVALS_MAX_TIME_DIST,
    PITCH_RANGE,
    PROGRAM_CHANGES,
    PROGRAMS,
    REMOVE_DUPLICATED_NOTES,
    SPECIAL_TOKENS,
    SUSTAIN_PEDAL_DURATION,
    TEMPO_RANGE,
    TIME_SIGNATURE_RANGE,
    USE_CHORDS,
    USE_PITCH_BENDS,
    USE_PITCH_INTERVALS,
    USE_PITCHDRUM_TOKENS,
    USE_PROGRAMS,
    USE_RESTS,
    USE_SUSTAIN_PEDALS,
    USE_TEMPOS,
    USE_TIME_SIGNATURE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

IGNORED_CONFIG_KEY_DICT = [
    "miditok_version",
    "symusic_version",
    "hf_tokenizers_version",
]


@dataclass
class Event:
    r"""
    Event class, representing a token and its characteristics.

    The type corresponds to the token type (e.g. *Pitch*, *Position*...) and its value.
    These two attributes are used to build its string representation (``__str__``),
    used in the Vocabulary class to map an event to its corresponding token.
    This class is mainly used during tokenization when the tokens / events have
    to be sorted by time.
    """

    type_: str
    value: str | int
    time: int | float = None
    program: int = None
    desc: Any = None

    def __str__(self) -> str:
        """
        Return the string value of the ``Event``.

        :return: string value of the ``Event`` as a combination of its type and value.
        """
        return f"{self.type_}_{self.value}"

    def __repr__(self) -> str:
        """
        Return the representation of this ``Event``.

        :return: representation of the event.
        """
        return (
            f"Event(type={self.type_}, value={self.value}, time={self.time},"
            f" desc={self.desc})"
        )


@dataclass
class TokSequence:
    r"""
    Sequence of token.

    A ``TokSequence`` can represent tokens by their several forms:
    * tokens (list of str): tokens as sequence of strings;
    * ids (list of int), these are the one to be fed to models;
    * events (list of Event): Event objects that can carry time or other information
    useful for debugging;
    * bytes (str): ids are converted into unique bytes, all joined together in a single
    string. This is used by MidiTok internally for BPE.

    Bytes are used internally by MidiTok for Byte Pair Encoding.
    The ``ids_are_bpe_encoded`` attribute tells if ``ids`` is encoded with BPE.

    :py:meth:`miditok.MIDITokenizer.complete_sequence`
    """

    tokens: list[str | list[str]] = None
    ids: list[int | list[int]] = None  # BPE can be applied on ids
    bytes: str = None  # noqa: A003
    events: list[Event | list[Event]] = None
    ids_bpe_encoded: bool = False
    _ids_no_bpe: list[int | list[int]] = None

    def __len__(self) -> int:
        """
        Return the length of the sequence.

        :return: number of elements in the sequence.
        """
        if self.ids is not None:
            return len(self.ids)
        if self.tokens is not None:
            return len(self.tokens)
        if self.events is not None:
            return len(self.events)
        if self.bytes is not None:
            return len(self.bytes)
        if self._ids_no_bpe is not None:
            return len(self._ids_no_bpe)

        msg = (
            "This TokSequence seems to not be initialized, all its attributes "
            "are None."
        )
        raise ValueError(msg)

    def __getitem__(self, idx: int) -> int | str | Event:
        """
        Return the ``idx``th element of the sequence.

        It checks by order: ids, tokens, events, bytes.

        :param idx: index of the element to retrieve.
        :return: ``idx``th element.
        """
        if self.ids is not None:
            return self.ids[idx]
        if self.tokens is not None:
            return self.tokens[idx]
        if self.events is not None:
            return self.events[idx]
        if self.bytes is not None:
            return self.bytes[idx]
        if self._ids_no_bpe is not None:
            return self._ids_no_bpe[idx]

        msg = (
            "This TokSequence seems to not be initialized, all its attributes "
            "are None."
        )
        raise ValueError(msg)

    def __eq__(self, other: TokSequence) -> bool:
        r"""
        Check that too sequences are equal.

        This is performed by comparing their attributes (ids, tokens...).
        **Both sequences must have at least one common attribute initialized (not None)
        for this method to work, otherwise it will return False.**.

        :param other: other sequence to compare.
        :return: ``True`` if the sequences have equal attributes.
        """
        # Start from True assumption as some attributes might be unfilled (None)
        attributes = ["tokens", "ids", "bytes", "events"]
        eq = [True for _ in attributes]
        one_common_attr = False
        for i, attr in enumerate(attributes):
            if getattr(self, attr) is not None and getattr(other, attr) is not None:
                eq[i] = getattr(self, attr) == getattr(other, attr)
                one_common_attr = True

        return one_common_attr and all(eq)


class TokenizerConfig:
    r"""
    Tokenizer configuration, to be used with all tokenizers.

    :param pitch_range: range of MIDI pitches to use. Pitches can take values between
        0 and 127 (included). The `General MIDI 2 (GM2) specifications
        <https://www.midi.org/specifications-old/item/general-midi-2>`_ indicate the
        **recommended** ranges of pitches per MIDI program (instrument). These
        recommended ranges can also be found in ``miditok.constants``. In all cases,
        the range from 21 to 108 (included) covers all the recommended values. When
        processing a MIDI, the notes with pitches under or above this range can be
        discarded. (default: ``(21, 109)``)
    :param beat_res: beat resolutions, as a dictionary in the form:
        ``{(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}``.
        The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar,
        and the values are the resolution (in samples per beat) to apply to the ranges,
        ex 8. This allows to use ``Duration`` / ``TimeShift`` tokens of different
        lengths / resolutions. Note: for tokenization with ``Position`` tokens, the
        total number of possible positions will be set at four times the maximum
        resolution given (``max(beat_res.values)``\).
        (default: ``{(0, 4): 8, (4, 12): 4}``)
    :param num_velocities: number of velocity bins. In the MIDI norm, velocities can
        take up to 128 values (0 to 127). This parameter allows to reduce the number
        of velocity values. The velocities of the MIDIs resolution will be downsampled
        to ``num_velocities`` values, equally separated between 0 and 127.
        (default: ``32``)
    :param special_tokens: list of special tokens. This must be given as a list of
        strings, that should represent either the token type alone (e.g. ``PAD``) or
        the token type and its value separated by an underscore (e.g. ``Genre_rock``).
        If two or more underscores are given, all but the last one will be replaced
        with dashes (-). (default: ``["PAD", "BOS", "EOS", "MASK"]``\)
    :param use_chords: will use ``Chord`` tokens, if the tokenizer is compatible. A
        ``Chord`` token indicates the presence of a chord at a certain time step.
        MidiTok uses a chord detection method based on onset times and duration. This
        allows MidiTok to detect precisely chords without ambiguity, whereas most chord
        detection methods in symbolic music based on chroma features can't. Note that
        using chords will increase the tokenization time, especially if you are working
        on music with a high "note density". (default: ``False``)
    :param use_rests: will use ``Rest`` tokens, if the tokenizer is compatible.
        ``Rest`` tokens will be placed whenever a portion of time is silent, i.e. no
        note is being played. This token type is decoded as a ``TimeShift`` event. You
        can choose the minimum and maximum rests values to represent with the
        ``beat_res_rest`` argument. (default: ``False``)
    :param use_tempos: will use ``Tempo`` tokens, if the tokenizer is compatible.
        ``Tempo`` tokens will specify the current tempo. This allows to train a model
        to predict tempo changes. Tempo values are quantized accordingly to the
        ``num_tempos`` and ``tempo_range`` entries in the ``additional_tokens``
        dictionary (default is 32 tempos from 40 to 250). (default: ``False``)
    :param use_time_signatures: will use ``TimeSignature`` tokens, if the tokenizer is
        compatible. ``TimeSignature`` tokens will specify the current time signature.
        Note that :ref:`REMI` adds a ``TimeSignature`` token at the beginning of each
        Bar (i.e. after ``Bar`` tokens), while :ref:`TSD` and :ref:`MIDI-Like` will
        only represent time signature changes (MIDI messages) as they come. If you want
        more "recalls" of the current time signature within your token sequences, you
        can preprocess your MIDI file to add more ``TimeSignatureChange`` objects.
        (default: ``False``)
    :param use_sustain_pedals: will use ``Pedal`` tokens to represent the sustain pedal
        events. In multitrack setting, The value of each ``Pedal`` token will be equal
        to the program of the track. (default: ``False``)
    :param use_pitch_bends: will use ``PitchBend`` tokens. In multitrack setting, a
        ``Program`` token will be added before each ``PitchBend` token.
        (default: ``False``)
    :param use_pitch_intervals: if given True, will represent the pitch of the notes
        with pitch intervals tokens. This way, successive and simultaneous notes will
        be represented with respectively ``PitchIntervalTime`` and
        ``PitchIntervalChord`` tokens. A good example is depicted in
        :ref:`Additional tokens`. This option is to be used with the
        ``max_pitch_interval`` and ``pitch_intervals_max_time_dist`` arguments.
        (default: False)
    :param use_programs: will use ``Program`` tokens to specify the instrument/MIDI
        program of the notes, if the tokenizer is compatible (:ref:`TSD`, :ref:`REMI`,
        :ref:`MIDI-Like`, :ref:`Structured` and :ref:`CPWord`). Use this parameter with
        the ``programs``, ``one_token_stream_for_programs`` and `program_changes`
        arguments. By default, it will prepend a ``Program`` tokens before each
        ``Pitch``/``NoteOn`` token to indicate its associated instrument, and will
        treat all the tracks of a MIDI as a single sequence of tokens. :ref:`CPWord`,
        :ref:`Octuple` and :ref:`MuMIDI` add a ``Program`` tokens with the stacks of
        ``Pitch``, ``Velocity`` and ``Duration`` tokens. The :ref:`Octuple`, :ref:`MMM`
        and :ref:`MuMIDI` tokenizers use natively ``Program`` tokens, this option is
        always enabled. (default: ``False``)
    :param use_pitchdrum_tokens: will use dedicated ``PitchDrum`` tokens for pitches
        of drums tracks. In the MIDI norm, the pitches of drums tracks corresponds to
        discrete drum elements (bass drum, high tom, cymbals...) which are unrelated to
        the pitch value of other instruments/programs. Using dedicated tokens for drums
        allow to disambiguate this, and is thus recommended. (default: ``True``)
    :param beat_res_rest: the beat resolution of ``Rest`` tokens. It follows the same
        data pattern as the ``beat_res`` argument, however the maximum resolution for
        rests cannot be higher than the highest "global" resolution (``beat_res``).
        Rests are considered complementary to other time tokens (``TimeShift``, ``Bar``
        and ``Position``). If in a given situation, ``Rest`` tokens cannot represent
        time with the exact precision, other time times will complement them.
        (default: ``{(0, 1): 8, (1, 2): 4, (2, 12): 2}``)
    :param chord_maps: list of chord maps, to be given as a dictionary where keys are
        chord qualities (e.g. "maj") and values pitch maps as tuples of integers (e.g.
        ``(0, 4, 7)``). You can use ``miditok.constants.CHORD_MAPS`` as an example.
        (default: ``miditok.constants.CHORD_MAPS``)
    :param chord_tokens_with_root_note: to specify the root note of each chord in
        ``Chord`` tokens. Tokens will look like ``Chord_C:maj``. (default: ``False``)
    :param chord_unknown: range of number of notes to represent unknown chords. If you
        want to represent chords that does not match any combination in ``chord_maps``,
        use this argument. Leave ``None`` to not represent unknown chords.
        (default: ``None``)
    :param num_tempos: number of tempos "bins" to use. (default: ``32``)
    :param tempo_range: range of minimum and maximum tempos within which the bins fall.
        (default: ``(40, 250)``)
    :param log_tempos: will use log scaled tempo values instead of linearly scaled.
        (default: ``False``)
    :param remove_duplicated_notes: will remove duplicated notes before tokenizing
        MIDIs. Notes with the same onset time and pitch value will be deduplicated.
        This option will slightly increase the tokenization time. This option will add
        an extra note sorting step in the MIDI preprocessing, which can increase the
        overall tokenization time. (default: ``False``)
    :param delete_equal_successive_tempo_changes: setting this option True will delete
        identical successive tempo changes when preprocessing a MIDI file after loading
        it. For examples, if a MIDI has two tempo changes for tempo 120 at tick 1000
        and the next one is for tempo 121 at tick 1200, during preprocessing the tempo
        values are likely to be downsampled and become identical (120 or 121). If
        that's the case, the second tempo change will be deleted and not tokenized.
        This parameter doesn't apply for tokenizations that natively inject the tempo
        information at recurrent timings (e.g. :ref:`Octuple`). For others, note that
        setting it True might reduce the number of ``Tempo`` tokens and in turn the
        recurrence of this information. Leave it False if you want to have recurrent
        ``Tempo`` tokens, that you might inject yourself by adding ``TempoChange``
        objects to your MIDIs. (default: ``False``)
    :param time_signature_range: range as a dictionary
        ``{denom_i: [num_i1, ..., num_in]/(min_num_i, max_num_i)}``.
        (default: ``{8: [3, 12, 6], 4: [5, 6, 3, 2, 1, 4]}``)
    :param sustain_pedal_duration: by default, the tokenizer will use ``PedalOff``
        tokens to mark the offset times of pedals. By setting this parameter True, it
        will instead use ``Duration`` tokens to explicitly express their durations. If
        you use this parameter, make sure to configure ``beat_res`` to cover the
        durations you expect. (default: ``False``)
    :param pitch_bend_range: range of the pitch bend to consider, to be given as a
        tuple with the form ``(lowest_value, highest_value, num_of_values)``. There
        will be ``num_of_values`` tokens equally spaced between ``lowest_value` and
        `highest_value``. (default: ``(-8192, 8191, 32)``)
    :param delete_equal_successive_time_sig_changes: setting this option True will
        delete identical successive time signature changes when preprocessing a MIDI
        file after loading it. For examples, if a MIDI has two time signature changes
        for 4/4 at tick 1000 and the next one is also 4/4 at tick 1200, the second time
        signature change will be deleted and not tokenized. This parameter doesn't
        apply for tokenizations that natively inject the time signature information at
        recurrent timings (e.g. :ref:`Octuple`). For others, note that setting it
        ``True`` might reduce the number of ``TimeSig`` tokens and in turn the
        recurrence of this information. Leave it ``False`` if you want to have
        recurrent ``TimeSig`` tokens, that you might inject yourself by adding
        ``TimeSignatureChange`` objects to your MIDIs. (default: ``False``)
    :param programs: sequence of MIDI programs to use. Note that ``-1`` is used and
        reserved for drums tracks. (default: ``list(range(-1, 128))``, from -1 to 127
        included)
    :param one_token_stream_for_programs: when using programs (``use_programs``), this
        parameters will make the tokenizer treat all the tracks of a MIDI as a single
        stream of tokens. A ``Program`` token will prepend each ``Pitch``, ``NoteOn``
        and ``NoteOff`` tokens to indicate their associated program / instrument. Note
        that this parameter is always set to True for :ref:`MuMIDI` and :ref:`MMM`.
        Setting it to False will make the tokenizer not use ``Programs``, but will
        allow to still have ``Program`` tokens in the vocabulary. (default: ``True``)
    :param program_changes: to be used with ``use_programs``. If given True, the
        tokenizer will place ``Program`` tokens whenever a note is being played by an
        instrument different from the last one. This mimics the ProgramChange MIDI
        messages. If given False, a ``Program`` token will precede each note tokens
        instead. This parameter only apply for :ref:`REMI`, :ref:`TSD` and
        :ref:`MIDI-Like`. If you set it True while your tokenizer is not int
        ``one_token_stream`` mode, a ``Program`` token at the beginning of each track
        token sequence. (default: ``False``)
    :param max_pitch_interval: sets the maximum pitch interval that can be represented.
        (default: ``16``)
    :param pitch_intervals_max_time_dist: sets the default maximum time interval in
        beats between two consecutive notes to be represented with pitch intervals.
        (default: ``1``)
    :param drums_pitch_range: range of pitch values to use for the drums tracks. This
        argument is only used when ``use_drums_pitch_tokens`` is ``True``. (default:
        ``(27, 88)``, recommended range from the GM2 specs without the "Applause" at
        pitch 88 of the orchestra drum set)
    :param kwargs: additional parameters that will be saved in
        ``config.additional_params``.
    """

    def __init__(
        self,
        pitch_range: tuple[int, int] = PITCH_RANGE,
        beat_res: dict[tuple[int, int], int] = BEAT_RES,
        num_velocities: int = NUM_VELOCITIES,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        use_chords: bool = USE_CHORDS,
        use_rests: bool = USE_RESTS,
        use_tempos: bool = USE_TEMPOS,
        use_time_signatures: bool = USE_TIME_SIGNATURE,
        use_sustain_pedals: bool = USE_SUSTAIN_PEDALS,
        use_pitch_bends: bool = USE_PITCH_BENDS,
        use_programs: bool = USE_PROGRAMS,
        use_pitch_intervals: bool = USE_PITCH_INTERVALS,
        use_pitchdrum_tokens: bool = USE_PITCHDRUM_TOKENS,
        beat_res_rest: dict[tuple[int, int], int] = BEAT_RES_REST,
        chord_maps: dict[str, tuple] = CHORD_MAPS,
        chord_tokens_with_root_note: bool = CHORD_TOKENS_WITH_ROOT_NOTE,
        chord_unknown: tuple[int, int] = CHORD_UNKNOWN,
        num_tempos: int = NUM_TEMPOS,
        tempo_range: tuple[int, int] = TEMPO_RANGE,
        log_tempos: bool = LOG_TEMPOS,
        remove_duplicated_notes: bool = REMOVE_DUPLICATED_NOTES,
        delete_equal_successive_tempo_changes: bool = (
            DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES
        ),
        time_signature_range: dict[
            int, list[int] | tuple[int, int]
        ] = TIME_SIGNATURE_RANGE,
        sustain_pedal_duration: bool = SUSTAIN_PEDAL_DURATION,
        pitch_bend_range: tuple[int, int, int] = PITCH_BEND_RANGE,
        delete_equal_successive_time_sig_changes: bool = (
            DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES
        ),
        programs: Sequence[int] = PROGRAMS,
        one_token_stream_for_programs: bool = ONE_TOKEN_STREAM_FOR_PROGRAMS,
        program_changes: bool = PROGRAM_CHANGES,
        max_pitch_interval: int = MAX_PITCH_INTERVAL,
        pitch_intervals_max_time_dist: bool = PITCH_INTERVALS_MAX_TIME_DIST,
        drums_pitch_range: tuple[int, int] = DRUM_PITCH_RANGE,
        **kwargs,
    ) -> None:
        # Checks
        if not 0 <= pitch_range[0] < pitch_range[1] <= 127:
            msg = (
                "`pitch_range` must be within 0 and 127, and an first value "
                f"greater than the second (received {pitch_range})"
            )
            raise ValueError(msg)
        if not 1 <= num_velocities <= 127:
            msg = (
                "`num_velocities` must be within 1 and 127 (received "
                f"{num_velocities})"
            )
            raise ValueError(msg)
        if max_pitch_interval and not 0 <= max_pitch_interval <= 127:
            msg = (
                "`max_pitch_interval` must be within 0 and 127 (received "
                f"{max_pitch_interval})."
            )
            raise ValueError(msg)
        if use_time_signatures:
            for denominator in time_signature_range:
                if not log2(denominator).is_integer():
                    msg = (
                        "`time_signature_range` contains an invalid time signature "
                        "denominator. The MIDI norm only supports powers of 2"
                        f"denominators. Received {denominator}"
                    )
                    raise ValueError(msg)

        # Global parameters
        self.pitch_range: tuple[int, int] = pitch_range
        self.beat_res: dict[tuple[int, int], int] = beat_res
        self.num_velocities: int = num_velocities
        self.special_tokens: list[str] = []
        for special_token in special_tokens:
            parts = special_token.split("_")
            if len(parts) == 1:
                parts.append("None")
            elif len(parts) > 2:
                parts = ["-".join(parts[:-1]), parts[-1]]
                warnings.warn(
                    f"miditok.TokenizerConfig: special token {special_token} must"
                    " contain one underscore (_).This token will be saved as"
                    f" {'_'.join(parts)}.",
                    stacklevel=2,
                )
            self.special_tokens.append("_".join(parts))
        self.remove_duplicated_notes = remove_duplicated_notes

        # Additional token types params, enabling additional token types
        self.use_chords: bool = use_chords
        self.use_rests: bool = use_rests
        self.use_tempos: bool = use_tempos
        self.use_time_signatures: bool = use_time_signatures
        self.use_sustain_pedals: bool = use_sustain_pedals
        self.use_pitch_bends: bool = use_pitch_bends
        self.use_programs: bool = use_programs
        self.use_pitch_intervals = use_pitch_intervals
        self.use_pitchdrum_tokens = use_pitchdrum_tokens

        # Rest params
        self.beat_res_rest: dict[tuple[int, int], int] = beat_res_rest
        if self.use_rests:
            max_rest_res = max(self.beat_res_rest.values())
            max_global_res = max(self.beat_res.values())
            if max_rest_res > max_global_res:
                msg = (
                    "The maximum resolution of the rests must be inferior or equal to"
                    "the maximum resolution of the global beat resolution"
                    f"(``config.beat_res``). Expected <= {max_global_res},"
                    f"{max_rest_res} was given."
                )
                raise ValueError(msg)

        # Chord params
        self.chord_maps: dict[str, tuple] = chord_maps
        # Tokens will look as "Chord_C:maj"
        self.chord_tokens_with_root_note: bool = chord_tokens_with_root_note
        # (3, 6) for chords between 3 and 5 notes
        self.chord_unknown: tuple[int, int] = chord_unknown

        # Tempo params
        self.num_tempos: int = num_tempos
        self.tempo_range: tuple[int, int] = tempo_range  # (min_tempo, max_tempo)
        self.log_tempos: bool = log_tempos
        self.delete_equal_successive_tempo_changes = (
            delete_equal_successive_tempo_changes
        )

        # Time signature params
        self.time_signature_range = {
            denominator: (
                list(range(numerators[0], numerators[1] + 1))
                if isinstance(numerators, tuple)
                else numerators
            )
            for denominator, numerators in time_signature_range.items()
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
        self.program_changes = program_changes and use_programs

        # Pitch as interval tokens
        self.max_pitch_interval = max_pitch_interval
        self.pitch_intervals_max_time_dist = pitch_intervals_max_time_dist

        # Drums
        self.drums_pitch_range = drums_pitch_range

        # Pop legacy kwargs
        legacy_args = (
            ("nb_velocities", "num_velocities"),
            ("nb_tempos", "num_tempos"),
        )
        for legacy_arg, new_arg in legacy_args:
            if legacy_arg in kwargs:
                setattr(self, new_arg, kwargs.pop(legacy_arg))
                warnings.warn(
                    f"Argument {legacy_arg} has been renamed {new_arg}, you should"
                    " consider to updateyour code with this new argument name.",
                    stacklevel=2,
                )

        # Additional params
        self.additional_params = kwargs

    @property
    def max_num_pos_per_beat(self) -> int:
        """
        Returns the maximum number of positions per ticks covered by the config.

        :return: maximum number of positions per ticks covered by the config.
        """
        return max(self.beat_res.values())

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any], **kwargs) -> TokenizerConfig:
        r"""
        Instantiate an ``TokenizerConfig`` from a Python dictionary.

        :param input_dict: Dictionary that will be used to instantiate the
            configuration object.
        :param kwargs: Additional parameters from which to initialize the
            configuration object.
        :returns: The ``TokenizerConfig`` object instantiated from those parameters.
        """
        input_dict.update(**input_dict["additional_params"])
        input_dict.pop("additional_params")
        for key in IGNORED_CONFIG_KEY_DICT:
            if key in input_dict:
                input_dict.pop(key)
        return cls(**input_dict, **kwargs)

    def to_dict(self, serialize: bool = False) -> dict[str, Any]:
        r"""
        Serialize this configuration to a Python dictionary.

        :param serialize: will serialize the dictionary before returning it, so it can
            be saved to a JSON file.
        :return: Dictionary of all the attributes that make up this configuration
            instance.
        """
        dict_config = deepcopy(self.__dict__)
        if serialize:
            self.__serialize_dict(dict_config)
        return dict_config

    def __serialize_dict(self, dict_: dict) -> None:
        r"""
        Convert numpy arrays to lists recursively within a dictionary.

        :param dict_: dictionary to serialize
        """
        for key in dict_:
            if isinstance(dict_[key], dict):
                self.__serialize_dict(dict_[key])
            elif isinstance(dict_[key], ndarray):
                dict_[key] = dict_[key].tolist()

    def save_to_json(self, out_path: str | Path) -> None:
        r"""
        Save a tokenizer configuration object to the `out_path` path.

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
        dict_config["miditok_version"] = CURRENT_MIDITOK_VERSION
        dict_config["symusic_version"] = CURRENT_SYMUSIC_VERSION
        dict_config["hf_tokenizers_version"] = CURRENT_TOKENIZERS_VERSION

        with out_path.open("w") as outfile:
            json.dump(dict_config, outfile, indent=4)

    @classmethod
    def load_from_json(cls, config_file_path: str | Path) -> TokenizerConfig:
        r"""
        Load a tokenizer configuration from the `config_path` path.

        :param config_file_path: path to the configuration JSON file to load.
        """
        if isinstance(config_file_path, str):
            config_file_path = Path(config_file_path)

        with config_file_path.open() as param_file:
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

    def __eq__(self, other: TokenizerConfig) -> bool:
        """
        Check two configs are equal.

        :param other: other config object to compare.
        :return: `True` if all attributes are equal, `False` otherwise.
        """
        # We don't use the == operator as it yields False when comparing lists and
        # tuples containing the same elements. This method is not recursive and only
        # checks the first level of iterable values / attributes
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
