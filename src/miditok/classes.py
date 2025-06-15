"""Common classes."""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from numpy import ndarray

from .constants import (
    AC_NOTE_DENSITY_BAR,
    AC_NOTE_DENSITY_BAR_MAX,
    AC_NOTE_DENSITY_TRACK,
    AC_NOTE_DENSITY_TRACK_MAX,
    AC_NOTE_DENSITY_TRACK_MIN,
    AC_NOTE_DURATION_BAR,
    AC_NOTE_DURATION_TRACK,
    AC_PITCH_CLASS_BAR,
    AC_POLYPHONY_BAR,
    AC_POLYPHONY_MAX,
    AC_POLYPHONY_MIN,
    AC_POLYPHONY_TRACK,
    AC_REPETITION_TRACK,
    AC_REPETITION_TRACK_NUM_BINS,
    AC_REPETITION_TRACK_NUM_CONSEC_BARS,
    BEAT_RES,
    BEAT_RES_REST,
    CHORD_MAPS,
    CHORD_TOKENS_WITH_ROOT_NOTE,
    CHORD_UNKNOWN,
    DEFAULT_NOTE_DURATION,
    DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES,
    DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES,
    DRUM_PITCH_RANGE,
    ENCODE_IDS_SPLIT,
    LOG_TEMPOS,
    MANDATORY_SPECIAL_TOKENS,
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
    USE_NOTE_DURATION_PROGRAMS,
    USE_PITCH_BENDS,
    USE_PITCH_INTERVALS,
    USE_PITCHDRUM_TOKENS,
    USE_PROGRAMS,
    USE_RESTS,
    USE_SUSTAIN_PEDALS,
    USE_TEMPOS,
    USE_TIME_SIGNATURE,
    USE_VELOCITIES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    time: int = -1
    program: int = 0
    desc: str | int = 0

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
    string. This is used internally by MidiTok for the tokenizer's model (BPE, Unigram,
    WordPiece).

    Bytes are used internally by MidiTok for Byte Pair Encoding.
    The ``are_ids_encoded`` attribute tells if ``ids`` is encoded.

    :py:meth:`miditok.MusicTokenizer.complete_sequence` can be used to complete the
    non-initialized attributes.
    """

    tokens: list[str | list[str]] = field(default_factory=list)
    ids: list[int | list[int]] = field(default_factory=list)
    bytes: str = field(default_factory=str)
    events: list[Event | list[Event]] = field(default_factory=list)
    are_ids_encoded: bool = False
    _ticks_bars: list[int] = field(default_factory=list)  # slice/add not handled
    _ticks_beats: list[int] = field(default_factory=list)  # slice/add not handled
    _ids_decoded: list[int | list[int]] = field(default_factory=list)

    def split_per_bars(self) -> list[TokSequence]:
        """
        Split the sequence into subsequences corresponding to each bar.

        The method can only be called from sequences properly tokenized, otherwise it
        will throw an error.

        :return: list of subsequences for each bar.
        """
        return self._split_per_ticks(self._ticks_bars)

    def split_per_beats(self) -> list[TokSequence]:
        """
        Split the sequence into subsequences corresponding to each beat.

        The method can only be called from sequences properly tokenized, otherwise it
        will throw an error.

        :return: list of subsequences for each beat.
        """
        return self._split_per_ticks(self._ticks_beats)

    def _split_per_ticks(self, ticks: Sequence[int]) -> list[TokSequence]:
        idxs = [0]
        ti_prev = 0
        for bi in range(1, len(ticks)):
            ti = ti_prev
            while ti < len(self.events) and self.events[ti].time < ticks[bi]:
                ti += 1
            if ti == len(self.events):
                break
            idxs.append(ti)
            ti_prev = ti

        # Split the sequence
        idxs.append(None)
        subsequences = [self[idxs[i - 1] : idxs[i]] for i in range(1, len(idxs))]
        # Remove their _ticks_bars and _ticks_beats
        for subseq in subsequences:
            subseq._ticks_bars = subseq._ticks_beats = None

        return subsequences

    def __len__(self) -> int:
        """
        Return the length of the sequence.

        :return: number of elements in the sequence.
        """
        for attr_ in ("ids", "tokens", "events", "bytes"):
            if (length := len(getattr(self, attr_))) != 0:
                return length
        # Are all 0s
        return 0

    def __getitem__(self, val: int | slice) -> int | str | Event | TokSequence:
        """
        Return the ``idx``th element or slice of the sequence.

        If an integer is providing, it checks by order: ids, tokens, events, bytes.

        :param val: index of the element to retrieve.
        :return: ``idx``th element.
        """
        if isinstance(val, slice):
            return self.__slice(val)

        attr_to_check = ("ids", "tokens", "events", "bytes")
        for attr_ in attr_to_check:
            if len(getattr(self, attr_)) > 0:
                return getattr(self, attr_)[val]

        msg = (
            "This TokSequence seems to not be initialized, all its attributes are None."
        )
        raise ValueError(msg)

    def __slice(self, sli: slice) -> TokSequence:
        """
        Slice the ``TokSequence``.

        :param sli: slice object.
        :return: the slice of the self ``TokSequence``.
        """
        seq = replace(self)
        attributes = ["tokens", "ids", "bytes", "events", "_ids_decoded"]
        for attr in attributes:
            if len(getattr(self, attr)) > 0:
                setattr(seq, attr, getattr(self, attr)[sli])
        return seq

    def __eq__(self, other: object) -> bool:
        r"""
        Check that too sequences are equal.

        This is performed by comparing their attributes (ids, tokens...).
        **Both sequences must have at least one common attribute initialized (not None)
        for this method to work, otherwise it will return False.**.

        :param other: other sequence to compare.
        :return: ``True`` if the sequences have equal attributes.
        """
        if not isinstance(other, TokSequence):
            return False
        # Start from True assumption as some attributes might be unfilled (None)
        attributes = ["tokens", "ids", "bytes", "events"]
        eq = [True for _ in attributes]
        one_common_attr = False
        for i, attr in enumerate(attributes):
            if len(getattr(self, attr)) > 0 and len(getattr(other, attr)) > 0:
                eq[i] = getattr(self, attr) == getattr(other, attr)
                one_common_attr = True

        return one_common_attr and all(eq)

    def __add__(self, other: TokSequence) -> TokSequence:
        """
        Concatenate two ``TokSequence`` objects.

        The `ìds``, ``tokens``, ``events`` and ``bytes`` will be concatenated.

        :param other: other ``TokSequence``.
        :return: the two sequences concatenated.
        """
        seq = replace(self)
        seq += other
        return seq

    def __iadd__(self, other: TokSequence) -> TokSequence:
        """
        Concatenate the self ``TokSequence`` to another one.

        The `ìds``, ``tokens``, ``events`` and ``bytes`` will be concatenated.

        :param other: other ``TokSequence``.
        :return: the two sequences concatenated.
        """
        if not isinstance(other, TokSequence):
            msg = (
                "Addition to a `TokSequence` object can only be performed with other"
                f"`TokSequence` objects. Received: {other.__class__.__name__}"
            )
            raise ValueError(msg)
        attributes = ["tokens", "ids", "bytes", "events", "_ids_decoded"]
        for attr in attributes:
            self_attr, other_attr = getattr(self, attr), getattr(other, attr)
            setattr(self, attr, self_attr + other_attr)

        return self

    def __radd__(self, other: TokSequence) -> TokSequence:
        """
        Reverse addition operation, allowing ``TokSequence``s to be summed.

        :param other: other ``TokSequence``.
        :return: the two sequences concatenated.
        """
        if other == 0:
            return self
        return self.__add__(other)


def _format_special_token(token: str) -> str:
    """
    Format a special token provided by a user.

    The method will split it in a "type" and a "value" categories separated by an
    underscore.

    :param token: special token as string.
    :return: formated special token.
    """
    parts = token.split("_")
    if len(parts) == 1:
        parts.append("None")
    elif len(parts) > 2:
        parts = ["-".join(parts[:-1]), parts[-1]]
        warnings.warn(
            f"miditok.TokenizerConfig: special token {token} must"
            " contain one underscore (_).This token will be saved as"
            f" {'_'.join(parts)}.",
            stacklevel=2,
        )
    return "_".join(parts)


class TokenizerConfig:
    r"""
    Tokenizer configuration, to be used with all tokenizers.

    :param pitch_range: range of note pitches to use. Pitches can take values between
        0 and 127 (included). The `General MIDI 2 (GM2) specifications
        <https://www.midi.org/specifications-old/item/general-midi-2>`_ indicate the
        **recommended** ranges of pitches per MIDI program (instrument). These
        recommended ranges can also be found in ``miditok.constants``. In all cases,
        the range from 21 to 108 (included) covers all the recommended values. When
        processing a MIDI file, the notes with pitches under or above this range can be
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
    :param num_velocities: number of velocity bins. In the MIDI protocol, velocities can
        take up to 128 values (0 to 127). This parameter allows to reduce the number
        of velocity values. The velocities of the file will be downsampled to
        ``num_velocities`` values, equally spaced between 0 and 127. (default: ``32``)
    :param special_tokens: list of special tokens. The "PAD" token is required and
        will be included in the vocabulary anyway if you did not include it in
        ``special_tokens``. This must be given as a list of strings, that should
        represent either the token type alone (e.g. ``PAD``) or the token type and its
        value separated by an underscore (e.g. ``Genre_rock``). If two or more
        underscores are given, all but the last one will be replaced with dashes (-).
        (default: ``["PAD", "BOS", "EOS", "MASK"]``\)
    :param encode_ids_split: allows to split the token ids before encoding them with the
        tokenizer's model (BPE, Unigram, WordPiece), similarly to how words are split
        with spaces in text. Doing so, the tokenizer will learn tokens that represent
        note/musical events successions only occurring within bars or beats. Possible
        values for this argument are ``"bar"``, ``beat`` or ``no``. (default:
        ``bar``)
    :param use_velocities: whether to use ``Velocity`` tokens. The velocity is a feature
        from the MIDI protocol that corresponds to the force with which a note is
        played. This information can be measured by MIDI devices and brings information
        on the way a music is played. At playback, the velocity usually impacts the
        volume at which notes are played. If you are not using MIDI files, using
        velocity tokens will have no interest as the information is not present in the
        data and the default velocity value will be used. Not using velocity tokens
        allows to reduce the token sequence length. If you disable velocity tokens, the
        tokenizer will set the velocity of notes decoded from tokens to 100 by default.
        (default: ``True``)
    :param use_note_duration_programs: list of the MIDI programs (i.e. instruments) for
        which the note durations are be tokenized. The durations of the notes of the
        tracks with these programs will be tokenized as ``Duration_x.x.x`` tokens
        succeeding their associated ``Pitch``/``NoteOn`` tokens.
        **Note for rests:** ``Rest`` tokens are compatible when note `Duration` tokens
        are enabled. If you intend to use rests without enabling ``Program`` tokens
        (``use_programs``), this parameter should be left unchanged (i.e. using
        ``Duration`` tokens for all programs). If you intend to use rests while using
        ``Program`` tokens, all the programs in this parameter should also be in the
        ``programs`` parameter. (default: all programs from -1 (drums) to 127 included)
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
        only represent time signature changes as they come. If you want more "recalls"
        of the current time signature within your token sequences, you can preprocess
        a ``symusic.Score`` object to add more ``symusic.TimeSignature`` objects.
        (default: ``False``)
    :param use_sustain_pedals: will use ``Pedal`` tokens to represent the sustain pedal
        events. In multitrack setting, The value of each ``Pedal`` token will be equal
        to the program of the track. (default: ``False``)
    :param use_pitch_bends: will use ``PitchBend`` tokens. In multitrack setting, a
        ``Program`` token will be added before each ``PitchBend`` token.
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
        the ``programs``, ``one_token_stream_for_programs`` and ``program_changes``
        arguments. By default, it will prepend a ``Program`` tokens before each
        ``Pitch``/``NoteOn`` token to indicate its associated instrument, and will
        treat all the tracks of a file as a single sequence of tokens. :ref:`CPWord`,
        :ref:`Octuple` and :ref:`MuMIDI` add a ``Program`` tokens with the stacks of
        ``Pitch``, ``Velocity`` and ``Duration`` tokens. The :ref:`Octuple`, :ref:`MMM`
        and :ref:`MuMIDI` tokenizers use natively ``Program`` tokens, this option is
        always enabled. (default: ``False``)
    :param use_pitchdrum_tokens: will use dedicated ``PitchDrum`` tokens for pitches
        of drums tracks. In the MIDI protocol, the pitches of drums tracks corresponds
        to discrete drum elements (bass drum, high tom, cymbals...) which are unrelated
        to the pitch value of other instruments/programs. Using dedicated tokens for
        drums allow to disambiguate this, and is thus recommended. (default: ``True``)
    :param default_note_duration: default duration in beats to set for notes for which
        the duration is not tokenized. This parameter is used when decoding tokens to
        set the duration value of notes within tracks with programs not in the
        ``use_note_duration_programs`` configuration parameter. (default: ``0.5``)
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
    :param remove_duplicated_notes: will remove duplicated notes before tokenizing.
        Notes with the same onset time and pitch value will be deduplicated.
        This option will slightly increase the tokenization time. This option will add
        an extra note sorting step in the music file preprocessing, which can increase
        the overall tokenization time. (default: ``False``)
    :param delete_equal_successive_tempo_changes: setting this option True will delete
        identical successive tempo changes when preprocessing a music file after loading
        it. For examples, if a file has two tempo changes for tempo 120 at tick 1000
        and the next one is for tempo 121 at tick 1200, during preprocessing the tempo
        values are likely to be downsampled and become identical (120 or 121). If
        that's the case, the second tempo change will be deleted and not tokenized.
        This parameter doesn't apply for tokenizations that natively inject the tempo
        information at recurrent timings (e.g. :ref:`Octuple`). For others, note that
        setting it True might reduce the number of ``Tempo`` tokens and in turn the
        recurrence of this information. Leave it False if you want to have recurrent
        ``Tempo`` tokens, that you might inject yourself by adding ``symusic.Tempo``
        objects to a ``symusic.Score``. (default: ``False``)
    :param time_signature_range: range as a dictionary. They keys are denominators
        (beat/note value), the values can be either the list of associated numerators
        (``{denom_i: [num_i_1, ..., num_i_n]}``) or a tuple ranging from the minimum
        numerator to the maximum (``{denom_i: (min_num_i, max_num_i)}``).
        (default: ``{8: [3, 12, 6], 4: [5, 6, 3, 2, 1, 4]}``)
    :param sustain_pedal_duration: by default, the tokenizer will use ``PedalOff``
        tokens to mark the offset times of pedals. By setting this parameter True, it
        will instead use ``Duration`` tokens to explicitly express their durations. If
        you use this parameter, make sure to configure ``beat_res`` to cover the
        durations you expect. (default: ``False``)
    :param pitch_bend_range: range of the pitch bend to consider, to be given as a
        tuple with the form ``(lowest_value, highest_value, num_of_values)``. There
        will be ``num_of_values`` tokens equally spaced between ``lowest_value`` and
        ``highest_value``. (default: ``(-8192, 8191, 32)``)
    :param delete_equal_successive_time_sig_changes: setting this option True will
        delete identical successive time signature changes when preprocessing a music
        file after loading it. For examples, if a file has two time signature changes
        for 4/4 at tick 1000 and the next one is also 4/4 at tick 1200, the second time
        signature change will be deleted and not tokenized. This parameter doesn't
        apply for tokenizations that natively inject the time signature information at
        recurrent timings (e.g. :ref:`Octuple`). For others, note that setting it
        ``True`` might reduce the number of ``TimeSig`` tokens and in turn the
        recurrence of this information. Leave it ``False`` if you want to have
        recurrent ``TimeSig`` tokens, that you might inject yourself by adding
        ``symusic.TimeSignature`` objects to a ``symusic.Score``. (default: ``False``)
    :param programs: sequence of MIDI programs to use. ``-1`` is used and reserved for
        drums tracks. If ``use_programs`` is enabled, the tracks with programs outside
        of this list will be ignored during tokenization.
        (default: ``list(range(-1, 128))``, from -1 to 127 included)
    :param one_token_stream_for_programs: when using programs (``use_programs``), this
        parameters will make the tokenizer serialize all the tracks of a
        ``symusic.Score`` in a single sequence of tokens. A ``Program`` token will
        prepend each ``Pitch``, ``NoteOn`` and ``NoteOff`` tokens to indicate their
        associated program / instrument. Note that this parameter is always set to True
        for :ref:`MuMIDI`. If disabled, the tokenizer will still use ``Program`` tokens
        but will tokenize each track independently. (default: ``True``)
    :param program_changes: to be used with ``use_programs``. If given ``True``, the
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
    :param ac_polyphony_track: enables track-level polyphony attribute control tokens
        using :class:`miditok.attribute_controls.TrackOnsetPolyphony`. (default:
        ``False``).
    :param ac_polyphony_bar: enables bar-level polyphony attribute control tokens
        using :class:`miditok.attribute_controls.BarOnsetPolyphony`. (default:
        ``False``).
    :param ac_polyphony_min: minimum number of simultaneous notes for polyphony
        attribute control. (default: ``1``)
    :param ac_polyphony_max: maximum number of simultaneous notes for polyphony
        attribute control. (default: ``6``)
    :param ac_pitch_class_bar: enables bar-level pitch class attribute control tokens
        using :class:`miditok.attribute_controls.BarPitchClass`. (default: ``False``).
    :param ac_note_density_track: enables track-level note density attribute control
        tokens using :class:`miditok.attribute_controls.TrackNoteDensity`. (default:
        ``False``).
    :param ac_note_density_track_min: minimum note density per bar to consider.
        (default: ``0``)
    :param ac_note_density_track_max: maximum note density per bar to consider.
        (default: ``18``)
    :param ac_note_density_bar: enables bar-level note density attribute control
        tokens using :class:`miditok.attribute_controls.BarNoteDensity`. (default:
        ``False``).
    :param ac_note_density_bar_max: maximum note density per bar to consider.
        (default: ``18``)
    :param ac_note_duration_bar: enables bar-level note duration attribute control
        tokens using :class:`miditok.attribute_controls.BarNoteDuration`. (default:
        ``False``).
    :param ac_note_duration_track: enables track-level note duration attribute control
        tokens using :class:`miditok.attribute_controls.TrackNoteDuration`. (default:
        ``False``).
    :param ac_repetition_track: enables track-level repetition attribute control tokens
        using :class:`miditok.attribute_controls.TrackRepetition`. (default: ``False``).
    :param ac_repetition_track_num_bins: number of levels of repetitions. (default:
        ``10``)
    :param ac_repetition_track_num_consec_bars: number of successive bars to
        compare the repetition similarity between bars. (default: ``4``)
    :param kwargs: additional parameters that will be saved in
        ``config.additional_params``.
    """

    def __init__(
        self,
        pitch_range: tuple[int, int] = PITCH_RANGE,
        beat_res: dict[tuple[int, int], int] = BEAT_RES,
        num_velocities: int = NUM_VELOCITIES,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        encode_ids_split: Literal["bar", "beat", "no"] = ENCODE_IDS_SPLIT,
        use_velocities: bool = USE_VELOCITIES,
        use_note_duration_programs: Sequence[int] = USE_NOTE_DURATION_PROGRAMS,
        use_chords: bool = USE_CHORDS,
        use_rests: bool = USE_RESTS,
        use_tempos: bool = USE_TEMPOS,
        use_time_signatures: bool = USE_TIME_SIGNATURE,
        use_sustain_pedals: bool = USE_SUSTAIN_PEDALS,
        use_pitch_bends: bool = USE_PITCH_BENDS,
        use_programs: bool = USE_PROGRAMS,
        use_pitch_intervals: bool = USE_PITCH_INTERVALS,
        use_pitchdrum_tokens: bool = USE_PITCHDRUM_TOKENS,
        default_note_duration: int | float = DEFAULT_NOTE_DURATION,
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
        time_signature_range: Mapping[
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
        pitch_intervals_max_time_dist: int | float = PITCH_INTERVALS_MAX_TIME_DIST,
        drums_pitch_range: tuple[int, int] = DRUM_PITCH_RANGE,
        ac_polyphony_track: bool = AC_POLYPHONY_TRACK,
        ac_polyphony_bar: bool = AC_POLYPHONY_BAR,
        ac_polyphony_min: int = AC_POLYPHONY_MIN,
        ac_polyphony_max: int = AC_POLYPHONY_MAX,
        ac_pitch_class_bar: bool = AC_PITCH_CLASS_BAR,
        ac_note_density_track: bool = AC_NOTE_DENSITY_TRACK,
        ac_note_density_track_min: int = AC_NOTE_DENSITY_TRACK_MIN,
        ac_note_density_track_max: int = AC_NOTE_DENSITY_TRACK_MAX,
        ac_note_density_bar: bool = AC_NOTE_DENSITY_BAR,
        ac_note_density_bar_max: int = AC_NOTE_DENSITY_BAR_MAX,
        ac_note_duration_bar: bool = AC_NOTE_DURATION_BAR,
        ac_note_duration_track: bool = AC_NOTE_DURATION_TRACK,
        ac_repetition_track: bool = AC_REPETITION_TRACK,
        ac_repetition_track_num_bins: int = AC_REPETITION_TRACK_NUM_BINS,
        ac_repetition_track_num_consec_bars: int = AC_REPETITION_TRACK_NUM_CONSEC_BARS,
        **kwargs,
    ) -> None:
        # Checks
        if not 0 <= pitch_range[0] < pitch_range[1] <= 127:
            msg = (
                "`pitch_range` must be within 0 and 127, and an first value "
                f"greater than the second (received {pitch_range})"
            )
            raise ValueError(msg)
        if not 0 <= drums_pitch_range[0] < drums_pitch_range[1] <= 127:
            msg = (
                "`drums_pitch_range` must be within 0 and 127, and an first value "
                f"greater than the second (received {drums_pitch_range})"
            )
            raise ValueError(msg)
        if not 1 <= num_velocities <= 127:
            msg = (
                f"`num_velocities` must be within 1 and 127 (received {num_velocities})"
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
                        "denominator. MidiTok only supports powers of 2 denominators, "
                        f"does the MIDI protocol. Received {denominator}."
                    )
                    raise ValueError(msg)

        # Global parameters
        self.pitch_range: tuple[int, int] = pitch_range
        self.beat_res: dict[tuple[int, int], int] = beat_res
        self.num_velocities: int = num_velocities
        self.remove_duplicated_notes = remove_duplicated_notes
        self.encode_ids_split = encode_ids_split

        # Special tokens
        self.special_tokens: list[str] = []
        for special_token in list(special_tokens):
            token = _format_special_token(special_token)
            if token not in self.special_tokens:
                self.special_tokens.append(token)
            else:
                warnings.warn(
                    f"The special token {token} is present twice in your configuration."
                    f" Skipping its duplicated occurrence.",
                    stacklevel=2,
                )
        # Mandatory special tokens, no warning here
        for special_token in MANDATORY_SPECIAL_TOKENS:
            token = _format_special_token(special_token)
            if token not in self.special_tokens:
                self.special_tokens.append(token)

        # Additional token types params, enabling additional token types
        self.use_velocities: bool = use_velocities
        self.use_note_duration_programs: set[int] = set(use_note_duration_programs)
        self.use_chords: bool = use_chords
        self.use_rests: bool = use_rests
        self.use_tempos: bool = use_tempos
        self.use_time_signatures: bool = use_time_signatures
        self.use_sustain_pedals: bool = use_sustain_pedals
        self.use_pitch_bends: bool = use_pitch_bends
        self.use_programs: bool = use_programs
        self.use_pitch_intervals: bool = use_pitch_intervals
        self.use_pitchdrum_tokens: bool = use_pitchdrum_tokens

        # Duration
        self.default_note_duration = default_note_duration

        # Programs
        self.programs: set[int] = set(programs)
        # These needs to be set to False if the tokenizer is not using programs
        self.one_token_stream_for_programs = (
            one_token_stream_for_programs and use_programs
        )
        self.program_changes = program_changes and use_programs

        # Check for rest compatibility with duration tokens
        if self.use_rests and len(self.use_note_duration_programs) < 129:
            msg = (
                "Disabling rests tokens. `Rest` tokens are compatible when note "
                "`Duration` tokens are enabled."
            )
            if not self.use_programs:
                self.use_rests = False
                warnings.warn(
                    msg + " Your configuration explicitly disable `Program` (allowing"
                    "to tokenize any track) while disabling note `Duration` "
                    "tokens for some programs.",
                    stacklevel=2,
                )

            elif any(p not in self.use_note_duration_programs for p in self.programs):
                self.use_rests = False
                warnings.warn(
                    msg + "You enabled `Program` tokens while disabling note duration "
                    " tokens for programs (`use_note_duration_programs`) outside "
                    "of the supported `programs`.",
                    stacklevel=2,
                )

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
        self.sustain_pedal_duration = sustain_pedal_duration and self.use_sustain_pedals

        # Pitch bend params
        self.pitch_bend_range = pitch_bend_range

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

        # Attribute controls
        self.ac_polyphony_track = ac_polyphony_track
        self.ac_polyphony_bar = ac_polyphony_bar
        self.ac_polyphony_min = ac_polyphony_min
        self.ac_polyphony_max = ac_polyphony_max
        self.ac_pitch_class_bar = ac_pitch_class_bar
        self.ac_note_density_track = ac_note_density_track
        self.ac_note_density_track_min = ac_note_density_track_min
        self.ac_note_density_track_max = ac_note_density_track_max
        self.ac_note_density_bar = ac_note_density_bar
        self.ac_note_density_bar_max = ac_note_density_bar_max
        self.ac_note_duration_bar = ac_note_duration_bar
        self.ac_note_duration_track = ac_note_duration_track
        self.ac_repetition_track = ac_repetition_track
        self.ac_repetition_track_num_bins = ac_repetition_track_num_bins
        self.ac_repetition_track_num_consec_bars = ac_repetition_track_num_consec_bars

        # Additional params
        self.additional_params = kwargs

    # Using dataclass overly complicates all the checks performed after init and reduces
    # the types flexibility (sequence etc...).
    # Freezing the class could be done, but special cases (MMM) should be handled.
    """def __setattr__(self, name, value):
        if getattr(self, "_is_frozen", False) and name != "_is_frozen":
            raise AttributeError(
                f"Cannot modify frozen instance of {self.__class__.__name__}"
            )
        super().__setattr__(name, value)

    def freeze(self):
        object.__setattr__(self, "_is_frozen", True)"""

    @property
    def max_num_pos_per_beat(self) -> int:
        """
        Returns the maximum number of positions per ticks covered by the config.

        :return: maximum number of positions per ticks covered by the config.
        """
        return max(self.beat_res.values())

    @property
    def using_note_duration_tokens(self) -> bool:
        """
        Return whether the configuration allows to use note duration tokens.

        :return: whether the configuration allows to use note duration tokens for at
            least one program.
        """
        return len(self.use_note_duration_programs) > 0

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
        input_dict.update(**input_dict.pop("additional_params"))
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
        Recursively convert non-json-serializable values of a dict to lists.

        :param dict_: dictionary to serialize
        """
        for key, value in dict_.items():
            if key in {"beat_res", "beat_res_rest"}:
                dict_[key] = {f"{k1}_{k2}": v for (k1, k2), v in value.items()}
            elif isinstance(value, dict):
                self.__serialize_dict(value)
            elif isinstance(value, ndarray):
                dict_[key] = value.tolist()
            elif isinstance(value, set):
                dict_[key] = list(value)

    def save_to_json(self, out_path: Path) -> None:
        r"""
        Save a tokenizer configuration as a JSON file.

        :param out_path: path to the output configuration JSON file.
        """
        if isinstance(out_path, str):
            out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dict_config = self.to_dict(serialize=True)

        with out_path.open("w") as outfile:
            json.dump(dict_config, outfile, indent=4)

    @classmethod
    def load_from_json(cls, config_file_path: Path) -> TokenizerConfig:
        r"""
        Load a tokenizer configuration from a JSON file.

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

    def copy(self) -> TokenizerConfig:
        """
        Copy the ``TokenizerConfig``.

        :return: a copy of the ``TokenizerConfig``.
        """
        return deepcopy(self)

    def __eq__(self, other: object) -> bool:
        """
        Check two configs are equal.

        :param other: other config object to compare.
        :return: ``True`` if all attributes are equal, ``False`` otherwise.
        """
        # We don't use the == operator as it yields False when comparing lists and
        # tuples containing the same elements. This method is not recursive and only
        # checks the first level of iterable values / attributes
        if not isinstance(other, TokenizerConfig):
            return False
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
