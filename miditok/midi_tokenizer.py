"""
MIDI encoding base class and methods
"""
import json
import math
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from huggingface_hub import ModelHubMixin as HFHubMixin
from huggingface_hub import hf_hub_download
from symusic import (
    ControlChange,
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TextMeta,
    TimeSignature,
    Track,
)
from symusic.core import (
    NoteTickList,
    PedalTickList,
    PitchBendTickList,
    ScoreTick,
    TempoTickList,
    TimeSignatureTickList,
)

try:
    from miditoolkit import MidiFile
except ImportError:
    MidiFile = None
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from .bpe_iterator import BPEIterator
from .classes import Event, TokenizerConfig, TokSequence
from .constants import (
    BOS_TOKEN_NAME,
    CHR_ID_START,
    CURRENT_MIDITOK_VERSION,
    CURRENT_SYMUSIC_VERSION,
    CURRENT_TOKENIZERS_VERSION,
    DEFAULT_TOKENIZER_FILE_NAME,
    EOS_TOKEN_NAME,
    MIDI_FILES_EXTENSIONS,
    MIDI_LOADING_EXCEPTION,
    PITCH_CLASSES,
    TEMPO,
    TIME_SIGNATURE,
    UNKNOWN_CHORD_PREFIX,
)
from .data_augmentation import data_augmentation_tokens
from .data_augmentation.data_augmentation import get_offsets
from .utils import (
    convert_ids_tensors_to_list,
    detect_chords,
    get_midi_programs,
    merge_same_program_tracks,
    remove_duplicated_notes,
)
from .utils.utils import np_get_closest


def convert_sequence_to_tokseq(
    tokenizer, input_seq, complete_seq: bool = True, decode_bpe: bool = True
) -> Union[TokSequence, List[TokSequence]]:
    r"""Converts a sequence into a :class:`miditok.TokSequence` or list of
    :class:`miditok.TokSequence` objects with the appropriate format of the tokenizer
    being used.

    :param tokenizer: tokenizer being used with the sequence.
    :param input_seq: sequence to convert. It can be a list of ids (integers), tokens
        (string) or events (Event). It can also be a Pytorch or TensorFlow tensor, or
        Numpy array representing ids.
    :param complete_seq: will complete the output sequence(s). (default: True)
    :param decode_bpe: if the input sequence contains ids, and that they contain BPE
        tokens, these tokens will be decoded. (default: True)
    :return:
    """
    # Deduce the type of data (ids/tokens/events)
    try:
        arg = ("ids", convert_ids_tensors_to_list(input_seq))
    except (AttributeError, ValueError, TypeError, IndexError):
        if isinstance(input_seq[0], str) or (
            isinstance(input_seq[0], list) and isinstance(input_seq[0][0], str)
        ):
            arg = ("tokens", input_seq)
        else:  # list of Event, but unlikely
            arg = ("events", input_seq)

    # Deduce nb of subscripts / dims
    nb_io_dims = len(tokenizer.io_format)
    nb_seq_dims = 1
    if len(arg[1]) > 0 and isinstance(arg[1][0], list):
        nb_seq_dims += 1
        if len(arg[1][0]) > 0 and isinstance(arg[1][0][0], list):
            nb_seq_dims += 1
        elif len(arg[1][0]) == 0 and nb_seq_dims == nb_io_dims - 1:
            # Special case where the sequence contains no tokens, we increment anyway
            nb_seq_dims += 1

    # Check the number of dimensions is good
    # In case of no one_token_stream and one dimension short --> unsqueeze
    if not tokenizer.one_token_stream and nb_seq_dims == nb_io_dims - 1:
        warnings.warn(
            f"The input sequence has one dimension less than expected ({nb_seq_dims}"
            f"instead of {nb_io_dims}). It is being unsqueezed to conform with the"
            f"tokenizer's i/o format ({tokenizer.io_format})",
            stacklevel=2,
        )
        arg = (arg[0], [arg[1]])

    elif nb_seq_dims != nb_io_dims:
        raise ValueError(
            f"The input sequence does not have the expected dimension "
            f"({nb_seq_dims} instead of {nb_io_dims})."
        )

    # Convert to TokSequence
    if not tokenizer.one_token_stream and nb_io_dims == nb_seq_dims:
        seq = []
        for obj in arg[1]:
            kwarg = {arg[0]: obj}
            seq.append(TokSequence(**kwarg))
            if not tokenizer.is_multi_voc and seq[-1].ids is not None:
                seq[-1].ids_bpe_encoded = tokenizer._are_ids_bpe_encoded(seq[-1].ids)
    else:  # 1 subscript, one_token_stream and no multi-voc
        kwarg = {arg[0]: arg[1]}
        seq = TokSequence(**kwarg)
        if not tokenizer.is_multi_voc:
            seq.ids_bpe_encoded = tokenizer._are_ids_bpe_encoded(seq.ids)

    # decode BPE and complete the output sequence(s) if requested
    if tokenizer.has_bpe and decode_bpe:
        tokenizer.decode_bpe(seq)
    if complete_seq:
        if isinstance(seq, TokSequence):
            tokenizer.complete_sequence(seq)
        else:
            for seq_ in seq:
                tokenizer.complete_sequence(seq_)

    return seq


def _in_as_seq(complete: bool = True, decode_bpe: bool = True):
    r"""Decorator creating if necessary and completing a :class:`miditok.TokSequence`
    object before that the function is called. This decorator is made to be used by the
    :py:meth:`miditok.MIDITokenizer.tokens_to_midi` method.

    :param complete: will complete the sequence, i.e. complete its ``ids`` , ``tokens``
        and ``events`` .
    :param decode_bpe: will decode BPE, if applicable. This step is performed before
        completing the sequence.
    """

    def decorator(function: Optional[Callable] = None):
        def wrapper(*args, **kwargs):
            tokenizer = args[0]
            seq = args[1]
            if not isinstance(seq, TokSequence) and not all(
                isinstance(seq_, TokSequence) for seq_ in seq
            ):
                seq = convert_sequence_to_tokseq(tokenizer, seq, complete, decode_bpe)
            else:
                if tokenizer.has_bpe and decode_bpe:
                    tokenizer.decode_bpe(seq)
                if complete:
                    if isinstance(seq, TokSequence):
                        tokenizer.complete_sequence(seq)
                    else:
                        for seq_ in seq:
                            tokenizer.complete_sequence(seq_)

            args = list(args)
            args[1] = seq
            return function(*args, **kwargs)

        return wrapper

    return decorator


def _out_as_complete_seq(function: Callable):
    r"""Decorator completing an output :class:`miditok.TokSequence` object."""

    def wrapper(*args, **kwargs):
        self = args[0]
        res = function(*args, **kwargs)
        self.complete_sequence(res)
        return res

    return wrapper


def miditoolkit_to_symusic(midi: MidiFile) -> Score:
    score = Score(midi.ticks_per_beat)

    # MIDI events (except key signature)
    for time_sig in midi.time_signature_changes:
        score.time_signatures.append(
            TimeSignature(time_sig.time, time_sig.numerator, time_sig.denominator)
        )
    for tempo in midi.tempo_changes:
        score.tempos.append(Tempo(tempo.time, tempo.tempo))
    for lyric in midi.lyrics:
        score.lyrics.append(TextMeta(lyric.time, lyric.text))
    for marker in midi.markers:
        score.markers.append(TextMeta(marker.time, marker.text))

    # Track events
    for inst in midi.instruments:
        track = Track(
            name=inst.name,
            program=inst.program,
            is_drum=inst.is_drum,
        )
        for note in inst.notes:
            track.notes.append(
                Note(note.start, note.duration, note.pitch, note.velocity)
            )
        for control in inst.control_changes:
            track.controls.append(
                ControlChange(control.time, control.number, control.value)
            )
        for pb in inst.pitch_bends:
            track.pitch_bends.append(PitchBend(pb.time, pb.pitch))
        for pedal in inst.pedals:
            track.pedals.append(Pedal(pedal.start, pedal.duration))
        score.tracks.append(track)

    return score


class MIDITokenizer(ABC, HFHubMixin):
    r"""MIDI tokenizer base class, containing common methods and attributes for all
    tokenizers.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.classes.TokenizerConfig` object.
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful if
        the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: Optional[Union[str, Path]] = None,
    ):
        # Initialize params
        self.config = deepcopy(tokenizer_config)
        # vocab of prime tokens, can be viewed as unique char / bytes
        self._vocab_base = {}
        # the other way, to decode id (int) -> token (str)
        self.__vocab_base_inv = {}
        # id (int) -> byte (str), as this might not be chr(id) after BPE training
        self._vocab_base_id_to_byte = {}
        # byte (str) -> token (str), for basic tokens
        self._vocab_base_byte_to_token = {}
        # byte(s) -> token(s), for faster BPE decoding
        self._vocab_bpe_bytes_to_tokens = {}
        self.has_bpe = False
        # Fast BPE tokenizer backed with 🤗tokenizers
        self._bpe_model = None
        # Used in _notes_to_events, especially MIDILike
        self._note_on_off = False
        # Determines how the tokenizer will handle multiple tracks: either each track
        # as a single independent token stream (False), or all the tracks as a single
        # token stream (True).
        self.one_token_stream = False

        # Loading params, or initializing them from args
        if params is not None:
            # Will overwrite self.config
            self._load_params(params)
        else:
            # If no TokenizerConfig is given, we falls back to the default parameters
            if self.config is None:
                self.config = TokenizerConfig()

        # Tweak the tokenizer's configuration and / or attributes before creating the
        # vocabulary. This method is intended to be overridden by inheriting tokenizer
        # classes
        self._tweak_config_before_creating_voc()

        # For internal use, new time division to apply when preprocessing MIDIs
        self._time_division = max(res for res in self.config.beat_res.values())

        # Set one_token_stream mode according to the config params
        if self.config.use_programs:
            self.one_token_stream = self.config.one_token_stream_for_programs

        # Init duration and velocity values
        self.durations = self.__create_durations_tuples()
        # [1:] so that there is no velocity_0
        self.velocities = np.linspace(
            0, 127, self.config.num_velocities + 1, dtype=np.intc
        )[1:]
        self._first_beat_res = next(iter(self.config.beat_res.values()))
        for beat_range, res in self.config.beat_res.items():
            if 0 in beat_range:
                self._first_beat_res = res
                break

        # Tempos
        # _DEFAULT_TEMPO is the closest one to 120 that the tokenizer supports
        self.tempos = np.zeros(1)
        self._DEFAULT_TEMPO = TEMPO
        if self.config.use_tempos:
            self.tempos = self.__create_tempos()
            self._DEFAULT_TEMPO = self.tempos[np.argmin(np.abs(self.tempos - TEMPO))]

        # Rests
        self.rests = []
        if self.config.use_rests:
            self.rests = self.__create_rests()

        # Time Signatures
        self.time_signatures = [TIME_SIGNATURE]
        if self.config.use_time_signatures:
            self.time_signatures = self.__create_time_signatures()

        # Pitch bends
        self.pitch_bends = np.zeros(1)
        if self.config.use_pitch_bends:
            self.pitch_bends = self.__create_pitch_bends()

        # Vocabulary and token types graph
        if (
            len(self.vocab) == 0
        ):  # in case it was not already loaded by load_params, such as with BPE
            self.__create_vocabulary()
        self.tokens_types_graph = self._create_token_types_graph()
        self._add_special_tokens_to_types_graph()
        self._token_types_indexes = {}
        self._update_token_types_indexes()

        # For internal use, Duration/TimeShift/Rest values of the tokenizer
        self._durations_ticks = np.array(
            [
                (beat * res + pos) * self._time_division // res
                for beat, pos, res in self.durations
            ]
        )
        self._rests_ticks = np.array(
            [
                (beat * res + pos) * self._time_division // res
                for beat, pos, res in self.rests
            ]
        )

    def _tweak_config_before_creating_voc(self):
        # called after setting the tokenizer's TokenizerConfig (.config). To be
        # customized by tokenizer classes.
        pass

    @property
    def vocab(
        self,
    ) -> Union[Dict[str, int], List[Dict[str, int]]]:  # token (str) to its id (int)
        """Get the base vocabulary, as a dictionary linking tokens (str) to their ids
        (int). The different (hidden / protected) vocabulary attributes of the class
        are:

        * ``._vocab_base`` : Dict[str: int] token -> id - Registers all known base
            tokens.
        * ``.__vocab_base_inv`` : Dict[int: str] id -> token - Inverse of
            ``._base_vocab`` , to go the other way.
        * ``._vocab_base_id_to_byte`` : Dict[int: str] id -> byte - Link ids to their
            associated unique bytes.
        * ``._vocab_base_byte_to_token`` : Dict[str: str] - similar as above but for
            tokens.
        * ``._vocab_bpe_bytes_to_tokens`` : Dict[str: List[str]] byte(s) -> token(s)
            used to decode BPE.
        * ``._bpe_model.get_vocab()`` : Dict[str: int] byte -> id - bpe model
            vocabulary, based on unique bytes

        Before training the tokenizer with BPE, bytes are obtained by running
        ``chr(id)`` . After training, if we did start from an empty vocabulary, some
        base tokens might be removed from ``._vocab_base`` , if they were never found
        in the training samples. The base vocabulary being changed, ``chr(id)`` would
        then bind to incorrect bytes (on which byte succession would not have been
        learned). We register the original id/token/byte association in
        ``._vocab_base_id_to_byte`` and ``._vocab_base_byte_to_token`` .

        :return: the base vocabulary.
        """
        return self._vocab_base

    @property
    def vocab_bpe(self) -> Union[None, Dict[str, int]]:  # byte (str) to its id (int)
        r"""Returns the vocabulary learnt with BPE.
        In case the tokenizer has not been trained with BPE, it returns None.

        :return: the BPE model's vocabulary.
        """
        if not self.has_bpe:
            return None
        else:
            return self._bpe_model.get_vocab()

    @property
    def special_tokens(self) -> List[str]:
        r"""Returns the vocabulary learnt with BPE.
        In case the tokenizer has not been trained with BPE, it returns None.

        :return: special tokens of the tokenizer
        """
        return self.config.special_tokens

    @property
    def special_tokens_ids(self) -> Sequence[int]:
        r"""Returns the vocabulary learnt with BPE.
        In case the tokenizer has not been trained with BPE, it returns None.

        :return: special tokens of the tokenizer
        """
        return [self[token] for token in self.special_tokens]

    @property
    def _min_rest(self) -> int:
        if not self.config.use_rests:
            return 0
        else:
            return int(self._rests_ticks[0])

    def preprocess_midi(self, midi: Score):
        r"""Pre-process (in place) a MIDI file to resample its time and events values
        before tokenizing it. Its notes attributes (times, pitches, velocities) will be
        downsampled and sorted, duplicated notes removed, as well as tempos. Empty
        tracks (with no note) will be removed from the MIDI object. Notes with pitches
        outside ``self.config.pitch_range`` will be deleted.

        :param midi: MIDI object to preprocess.
        """
        # Merge instruments of the same program / inst before preprocessing them
        # This allows to avoid potential duplicated notes in some multitrack settings
        if self.config.use_programs and self.one_token_stream:
            merge_same_program_tracks(midi.tracks)

        for t in range(len(midi.tracks) - 1, -1, -1):
            # quantize notes attributes
            self._resample_notes(midi.tracks[t].notes, midi.ticks_per_quarter)
            # sort notes
            midi.tracks[t].notes.sort(key=lambda x: (x.start, x.pitch, x.end))
            # remove possible duplicated notes
            remove_duplicated_notes(midi.tracks[t].notes)
            if len(midi.tracks[t].notes) == 0:
                del midi.tracks[t]
                continue

            # Quantize sustain pedal and pitch bend
            if self.config.use_sustain_pedals and len(midi.tracks[t].pedals) > 0:
                self._resample_sustain_pedals(
                    midi.tracks[t].pedals, midi.ticks_per_quarter
                )
            if self.config.use_pitch_bends and len(midi.tracks[t].pitch_bends) > 0:
                self._resample_pitch_bends(
                    midi.tracks[t].pitch_bends, midi.ticks_per_quarter
                )

        # Process tempo changes
        if self.config.use_tempos and len(midi.tempos) > 0:
            self._resample_tempos(midi.tempos, midi.ticks_per_quarter)

        # Process time signature changes
        if len(midi.time_signatures) == 0:  # can sometimes happen
            midi.time_signatures.append(
                TimeSignature(0, *TIME_SIGNATURE)
            )  # 4/4 by default in this case
        if self.config.use_time_signatures:
            self._resample_time_signatures(midi.time_signatures, midi.ticks_per_quarter)

        # We do not change key signature changes, markers and lyrics here as they are
        # not used by MidiTok (yet)

        # Set the new time division
        midi.ticks_per_quarter = self._time_division

    def _resample_notes(self, notes: NoteTickList, time_division: int):
        r"""Resamples the note attributes: their pitch, velocity, start and end values.
        The new time division will correspond to the maximum resolution value
        (``self.config.beat_res.values()``) in the config.
        Note durations will be clipped to the maximum duration that can be handled by
        the tokenizer. This is done to prevent having incorrect offset values when
        computing rests. Notes with pitches outside of self.pitch_range will be
        deleted.

        :param notes: notes to preprocess.
        :param time_division: time division of the MIDI being parsed.
        """
        resampling_factor = self._time_division / time_division
        max_duration_ticks = (
            max(tu[1] for tu in self.config.beat_res) * self._time_division
        )
        pitches = range(*self.config.pitch_range)

        # Gather times and velocity values in lists
        onsets_offsets, velocities = [], []
        i = 0
        while i < len(notes):
            if notes[i].pitch not in pitches:
                del notes[i]
                continue
            onsets_offsets.append([notes[i].time, notes[i].end])
            velocities.append(notes[i].velocity)
            i += 1

        # Resample time, remove 0 durations, find closest velocities
        onsets_offsets = np.rint(np.array(onsets_offsets) * resampling_factor).astype(
            np.intc
        )
        dur_zero = np.where(onsets_offsets[:, 0] == onsets_offsets[:, 1])[0]
        if len(dur_zero) > 0:
            onsets_offsets[dur_zero, np.ones_like(dur_zero)] += 1
        dur_excess = np.where(
            onsets_offsets[:, 1] - onsets_offsets[:, 0] > max_duration_ticks
        )[0]
        if len(dur_excess) > 0:
            new_offsets = (
                onsets_offsets[dur_excess, np.zeros_like(dur_excess)]
                + max_duration_ticks
            )
            for idx, new_offset in zip(dur_excess, new_offsets):
                onsets_offsets[idx, 1] = new_offset
        velocities = np_get_closest(self.velocities, velocities)

        # Apply new values
        for i, ((onset, offset), vel) in enumerate(zip(onsets_offsets, velocities)):
            notes[i].time = onset
            notes[i].duration = offset - onset
            notes[i].velocity = vel

    def _resample_tempos(self, tempos: TempoTickList, time_division: int):
        r"""Resamples the times and tempo values of tempo change events.
        Consecutive identical tempo changes will be removed if
        ``self.config.delete_equal_successive_tempo_changes`` is True.

        :param tempos: tempo changes to resample.
        :param time_division: time division of the MIDI being parsed
        """
        resampling_factor = self._time_division / time_division

        # If we delete the successive equal tempo changes, we need to sort them by time
        # Otherwise it is not required here as the tokens will be sorted by time
        if self.config.delete_equal_successive_tempo_changes:
            tempos.sort(key=lambda x: x.time)

        # Gather times and velocity values in lists
        times, values = [], []
        for tempo in tempos:
            times.append(tempo.time)
            values.append(tempo.tempo)

        # Resample time, find closest tempos
        times = np.rint(np.array(times) * resampling_factor).astype(np.intc)
        values = np_get_closest(self.tempos, values)

        # Find groups of tempos at the same onset ticks, equal consecutive ones
        if len(tempos) > 1:
            # Keep only last tempo change for groups with same tick
            idx_groups = np.split(
                np.arange(len(times)), np.where(np.diff(times) != 0)[0] + 1
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    for idx_to_del in reversed(idx_group[:-1]):
                        times = np.delete(times, idx_to_del)
                        values = np.delete(values, idx_to_del)
                        del tempos[idx_to_del]
            # Deduplicate successive tempo changes with same tempo value
            if self.config.delete_equal_successive_tempo_changes:
                idx_groups = np.split(
                    np.arange(len(values)), np.where(np.diff(values) != 0)[0] + 1
                )
                for idx_group in reversed(idx_groups):
                    if len(idx_group) > 1:
                        for idx_to_del in reversed(idx_group[1:]):
                            times = np.delete(times, idx_to_del)
                            values = np.delete(values, idx_to_del)
                            del tempos[idx_to_del]

        # Apply new values
        for time, val, tempo in zip(times, values, tempos):
            tempo.time = time
            tempo.tempo = val

    def _resample_time_signatures(
        self, time_sigs: TimeSignatureTickList, time_division: int
    ):
        r"""Resamples the time signature changes.
        There are not delayed to the next bar (anymore since v3.0.0).
        See MIDI 1.0 Detailed specifications, pages 54 - 56, for more information on
        delayed time signature messages.

        :param time_sigs: time signature changes to quantize.
        :param time_division: time division of the MIDI being parsed.
        """
        resampling_factor = self._time_division / time_division

        # If we delete the successive equal time signature changes, we need to sort
        # them by time, otherwise it is not required here as the tokens will be sorted
        # by time
        if self.config.delete_equal_successive_time_sig_changes:
            time_sigs.sort(key=lambda x: x.time)

        # Gathers times and velocity values in lists
        # Removes time sigs with a numerator or denominator equal to 0.
        times, values = [], []
        i = 0
        while i < len(time_sigs):
            if time_sigs[i].numerator == 0 or time_sigs[i].denominator == 0:
                del time_sigs[i]
                continue
            times.append(time_sigs[i].time)
            values.append([time_sigs[i].numerator, time_sigs[i].denominator])
            i += 1

        # Resample time, find closest tempos
        # TODO align time on bars?
        times = np.rint(np.array(times) * resampling_factor).astype(np.intc)
        values = np.array(values, dtype=np.short)

        # Find groups of time signatures at the same onset ticks, == consecutive ones
        if len(time_sigs) > 1:
            # Keep only last time signature change for groups with same tick
            idx_groups = np.split(
                np.arange(len(times)), np.where(np.diff(times) != 0)[0] + 1
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    for idx_to_del in reversed(idx_group[:-1]):
                        times = np.delete(times, idx_to_del)
                        values = np.delete(values, idx_to_del, axis=0)
                        del time_sigs[idx_to_del]
            # Deduplicate successive time signature changes with same value
            if self.config.delete_equal_successive_time_sig_changes:
                idx_groups = np.split(
                    np.arange(len(values)), np.where(np.diff(values) != 0)[0] + 1
                )
                for idx_group in reversed(idx_groups):
                    if len(idx_group) > 1:
                        for idx_to_del in reversed(idx_group[1:]):
                            times = np.delete(times, idx_to_del)
                            del time_sigs[idx_to_del]

        # Apply new values
        for time, time_sig in zip(times, time_sigs):
            time_sig.time = time

        """ticks_per_bar = MIDITokenizer._compute_ticks_per_bar(
            time_sigs[0], time_division
        )
        previous_tick = 0  # first time signature change is always at tick 0
        prev_ts = time_sigs[0]
        # If we delete the successive equal tempo changes, we need to sort them by time
        # Otherwise it is not required here as the tokens will be sorted by time
        if self.config.delete_equal_successive_time_sig_changes:
            time_sigs.sort(key=lambda x: x.time)

        i = 1
        while i < len(time_sigs):
            time_sig = time_sigs[i]

            if (
                self.config.delete_equal_successive_time_sig_changes
                and (
                    time_sig.numerator,
                    time_sig.denominator,
                )
                == (prev_ts.numerator, prev_ts.denominator)
                or time_sig.numerator == 0
                or time_sig.denominator == 0
            ):
                del time_sigs[i]
                continue

            # determine the current bar of time sig
            bar_offset, rest = divmod(time_sig.time - previous_tick, ticks_per_bar)
            if (
                rest > 0
            ):  # time sig doesn't happen on a new bar, we update it to the next bar
                bar_offset += 1
                time_sig.time = previous_tick + bar_offset * ticks_per_bar

            # Update values
            ticks_per_bar = MIDITokenizer._compute_ticks_per_bar(
                time_sig, time_division
            )

            # If the current time signature is now at the same time as the previous
            # one, we delete the previous
            if time_sig.time == previous_tick:
                del time_sigs[i - 1]
                continue

            previous_tick = time_sig.time
            prev_ts = time_sig
            i += 1"""

    def _resample_sustain_pedals(self, pedals: PedalTickList, time_division: int):
        r"""Resamples the sustain pedal events from a track. Their onset and offset
        times will be adjusted according to the time division of the tokenizer.

        :param pedals: sustain pedal events.
        :param time_division: time division of the MIDI being parsed.
        """
        resampling_factor = self._time_division / time_division
        onsets_offsets = [[pedal.time, pedal.end] for pedal in pedals]

        # Resample time, remove 0 durations
        onsets_offsets = np.rint(np.array(onsets_offsets) * resampling_factor).astype(
            np.intc
        )
        dur_zero = np.where(onsets_offsets[:, 0] == onsets_offsets[:, 1])
        if len(dur_zero) > 0:
            onsets_offsets[dur_zero, np.ones_like(dur_zero)] += 1

        # Apply new values
        for i, (onset, offset) in enumerate(onsets_offsets):
            pedals[i].time = onset
            pedals[i].duration = offset - onset

    def _resample_pitch_bends(self, pitch_bends: PitchBendTickList, time_division: int):
        r"""Resamples the pitch bend events from a track. Their onset and offset times
        will be adjusted according to the time division of the tokenizer. While being
        downsampled, overlapping pitch bends will be deduplicated by keeping the one
        having the highest absolute value at a given tick.

        :param pitch_bends: pitch bend events.
        :param time_division: time division of the MIDI being parsed.
        """
        resampling_factor = self._time_division / time_division

        # Gather times and velocity values in lists
        times, values = [], []
        for pitch_bend in pitch_bends:
            times.append(pitch_bend.time)
            values.append(pitch_bend.value)

        # Resample time, remove 0 durations
        times = np.rint(np.array(times) * resampling_factor).astype(np.intc)
        values = np_get_closest(self.pitch_bends, values)

        # Find groups of pitch bends at the same onset ticks, and keep the > abs values
        if len(pitch_bends) > 1:
            idx_groups = np.split(
                np.arange(len(times)), np.where(np.diff(times) != 0)[0] + 1
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    values_group = values[idx_group]
                    max_abs_idx = np.argmax(np.max(np.abs(values_group)))
                    values[idx_group[0]] = values_group[max_abs_idx]
                    for idx_to_del in reversed(idx_group[1:]):
                        times = np.delete(times, idx_to_del)
                        values = np.delete(values, idx_to_del)
                        del pitch_bends[idx_to_del]

        # Apply new values
        for i, (time, value) in enumerate(zip(times, values)):
            pitch_bends[i].time = time
            pitch_bends[i].value = value

    def _midi_to_tokens(self, midi: Score) -> Union[TokSequence, List[TokSequence]]:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        The workflow of this method is as follows: the events (*Pitch*, *Velocity*,
        *Tempo*, *TimeSignature*...) are gathered into a list, then the time events
        are added. If `one_token_stream` is true, all events of all tracks are
        treated all at once, otherwise the events of each track are treated
        independently.

        :param midi: the MIDI object to convert.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        # Create events list
        all_events = []
        if not self.one_token_stream:
            if len(midi.tracks) == 0:
                all_events.append([])
            else:
                all_events = [[] for _ in range(len(midi.tracks))]

        # Global events (Tempo, TimeSignature)
        global_events = self._create_midi_events(midi)
        if self.one_token_stream:
            all_events += global_events
        else:
            for i in range(len(all_events)):
                all_events[i] += global_events

        # Adds track tokens
        for ti, track in enumerate(midi.tracks):
            track_events = self._create_track_events(track)
            if self.one_token_stream:
                all_events += track_events
            else:
                if self.config.program_changes:
                    # ProgramNoteOff desc to make sure it appears before Pedals and
                    # everything else
                    track_events.insert(
                        0, Event("Program", track.program, 0, desc="ProgramNoteOff")
                    )
                all_events[ti] += track_events
                all_events[ti].sort(key=lambda x: (x.time, self.__order(x)))
        if self.one_token_stream:
            all_events.sort(key=lambda x: (x.time, self.__order(x)))
            # Add ProgramChange (named Program) tokens if requested
            if self.config.program_changes:
                self._add_program_change_events(all_events)

        # Add time events
        if self.one_token_stream:
            all_events = self._add_time_events(all_events)
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                all_events[i] = self._add_time_events(all_events[i])
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

    @staticmethod
    def __order(event: Event) -> int:
        """Internal method used to sort events (tokens) depending on their type or
        context of appearance. This is required, especially for multitrack
        one-token-stream situations where there can be several tokens appearing at
        the same moment (tick) from different tracks, that need to be sorted.

        :param event: event to determine priority.
        :return: priority as an int
        """
        # Global MIDI tokens first
        if event.type in ["Tempo", "TimeSig"]:
            return 0
        # Then NoteOff
        elif event.type == "NoteOff" or (
            event.type == "Program" and event.desc == "ProgramNoteOff"
        ):
            return 1
        # Then track effects
        elif event.type in ["Pedal", "PedalOff"] or (
            event.type == "Duration" and event.desc == "PedalDuration"
        ):
            return 2
        elif event.type == "PitchBend" or (
            event.type == "Program" and event.desc == "ProgramPitchBend"
        ):
            return 3
        elif event.type == "ControlChange":
            return 4
        # Track notes then
        else:
            return 10

    def _create_track_events(self, track: Track) -> List[Event]:
        r"""Extract the tokens / events of individual tracks: *Pitch*, *Velocity*,
        *Duration*, *NoteOn*, *NoteOff* and optionally *Chord*, from a track
        (``miditoolkit.Instrument``).

        :param track: MIDI track to convert.
        :return: sequence of corresponding Events
        """
        # Make sure the notes are sorted first by their onset (start) times, second by
        # pitch: notes.sort(key=lambda x: (x.start, x.pitch)) (done in midi_to_tokens)
        program = track.program if not track.is_drum else -1
        events = []
        note_token_name = "NoteOn" if self._note_on_off else "Pitch"
        max_time_interval = 0
        if self.config.use_pitch_intervals:
            max_time_interval = (
                self._time_division * self.config.pitch_intervals_max_time_dist
            )
        previous_note_onset = -max_time_interval - 1
        previous_pitch_onset = -128  # lowest at a given time
        previous_pitch_chord = -128  # for chord intervals

        # Add chords
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                self._time_division,
                chord_maps=self.config.chord_maps,
                program=program,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_nb_notes_range=self.config.chord_unknown,
            )
            for chord in chords:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event("Program", program, chord.time, program, "ProgramChord")
                    )
                events.append(chord)

        # Add sustain pedal
        if self.config.use_sustain_pedals:
            for pedal in track.pedals:
                # If not using programs, the default value is 0
                events.append(
                    Event(
                        "Pedal",
                        program if self.config.use_programs else 0,
                        pedal.time,
                        program,
                    )
                )
                # PedalOff or Duration
                if self.config.sustain_pedal_duration:
                    index = np.argmin(np.abs(self._durations_ticks - pedal.duration))
                    events.append(
                        Event(
                            "Duration",
                            ".".join(map(str, self.durations[index])),
                            pedal.time,
                            program,
                            "PedalDuration",
                        )
                    )
                else:
                    events.append(Event("PedalOff", program, pedal.end, program))

        # Add pitch bend
        if self.config.use_pitch_bends:
            for pitch_bend in track.pitch_bends:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event(
                            "Program",
                            program,
                            pitch_bend.time,
                            program,
                            "ProgramPitchBend",
                        )
                    )
                events.append(
                    Event("PitchBend", pitch_bend.value, pitch_bend.time, program)
                )

        # Control changes (in the future, and handle pedals redundancy)

        # Creates the Note On, Note Off and Velocity events
        for note in track.notes:
            # Program
            if self.config.use_programs and not self.config.program_changes:
                events.append(
                    Event(
                        type="Program",
                        value=program,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )

            # Pitch / interval
            add_absolute_pitch_token = True
            if self.config.use_pitch_intervals and not track.is_drum:
                if note.start != previous_note_onset:
                    if (
                        note.start - previous_note_onset <= max_time_interval
                        and abs(note.pitch - previous_pitch_onset)
                        <= self.config.max_pitch_interval
                    ):
                        events.append(
                            Event(
                                type="PitchIntervalTime",
                                value=note.pitch - previous_pitch_onset,
                                time=note.start,
                                program=program,
                                desc=note.end,
                            )
                        )
                        add_absolute_pitch_token = False
                    previous_pitch_onset = previous_pitch_chord = note.pitch
                else:  # same onset time
                    if (
                        abs(note.pitch - previous_pitch_chord)
                        <= self.config.max_pitch_interval
                    ):
                        events.append(
                            Event(
                                type="PitchIntervalChord",
                                value=note.pitch - previous_pitch_chord,
                                time=note.start,
                                program=program,
                                desc=note.end,
                            )
                        )
                        add_absolute_pitch_token = False
                    else:
                        # We update previous_pitch_onset as there might be a chord
                        # interval starting from the current note to the next one.
                        previous_pitch_onset = note.pitch
                    previous_pitch_chord = note.pitch
                previous_note_onset = note.start
            if add_absolute_pitch_token:
                events.append(
                    Event(
                        type=note_token_name,
                        value=note.pitch,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )

            # Velocity
            events.append(
                Event(
                    type="Velocity",
                    value=note.velocity,
                    time=note.start,
                    program=program,
                    desc=f"{note.velocity}",
                )
            )

            # Duration / NoteOff
            if self._note_on_off:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event(
                            type="Program",
                            value=program,
                            time=note.end,
                            program=program,
                            desc="ProgramNoteOff",
                        )
                    )
                events.append(
                    Event(
                        type="NoteOff",
                        value=note.pitch,
                        time=note.end,
                        program=program,
                        desc=note.end,
                    )
                )
            else:
                index = np.argmin(np.abs(self._durations_ticks - note.duration))
                events.append(
                    Event(
                        type="Duration",
                        value=".".join(map(str, self.durations[index])),
                        time=note.start,
                        program=program,
                        desc=f"{note.duration} ticks",
                    )
                )

        return events

    @staticmethod
    def _add_program_change_events(events: List[Event]):
        """Adds inplace Program tokens acting as Program Changes to a list of Events.

        :param events: Events to add Programs
        """
        previous_program = None
        previous_type = None
        program_change_events = []
        for ei, event in enumerate(events):
            if (
                event.program is not None
                and event.program != previous_program
                and event.type not in ["Pedal", "PedalOff"]
                and not (event.type == "Duration" and previous_type == "Pedal")
            ):
                previous_program = event.program
                program_change_events.append(
                    (ei, Event("Program", event.program, event.time))
                )
            previous_type = event.type

        for idx, event in reversed(program_change_events):
            events.insert(idx, event)

    def _create_midi_events(self, midi: Score) -> List[Event]:
        r"""Create the *global* MIDI additional tokens: `Tempo` and `TimeSignature`.

        :param midi: midi to extract the events from.
        :return: list of Events.
        """
        events = []

        # First adds time signature tokens if specified
        if self.config.use_time_signatures:
            for time_sig in midi.time_signatures:
                if (
                    time_sig.numerator,
                    time_sig.denominator,
                ) not in self.time_signatures:
                    warnings.warn(
                        f"The MIDI contains a time signature ({time_sig}) outside of"
                        f"those supported by the tokenizer ({self.time_signatures})."
                        "You should either discard this MIDI or support this time"
                        "signature, or alternatively deleting it however if you are "
                        "using a beat-based tokenizer (REMI) the bars will be"
                        "incorrectly detected.",
                        stacklevel=2,
                    )
                events.append(
                    Event(
                        type="TimeSig",
                        value=f"{time_sig.numerator}/" f"{time_sig.denominator}",
                        time=time_sig.time,
                    )
                )

        # Adds tempo events if specified
        if self.config.use_tempos:
            events += [
                Event(
                    type="Tempo",
                    value=round(tempo.tempo, 2),  # req to handle c++ values
                    time=tempo.time,
                    desc=tempo.tempo,
                )
                for tempo in midi.tempos
            ]

        return events

    def _add_time_events(self, events: List[Event]) -> List[Event]:
        r"""Internal method intended to be implemented by inheriting classes.
        It creates the time events from the list of global and track events, and as
        such the final token sequence.

        :param events: note events to complete.
        :return: the same events, with time events inserted.
        """
        raise NotImplementedError

    def midi_to_tokens(
        self,
        midi: Score,
        apply_bpe_if_possible: bool = True,
    ) -> Union[TokSequence, List[TokSequence]]:
        r"""Tokenizes a MIDI file.
        This method returns a list of :class:`miditok.TokSequence`.

        If you are implementing your own tokenization by subclassing this class,
        **override the ``_midi_to_tokens`` method**. This method implement necessary
        MIDI preprocessing.

        :param midi: the MIDI object to convert.
        :param apply_bpe_if_possible: will apply BPE if the tokenizer's vocabulary was
            learned with.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Tokenize it
        tokens = self._midi_to_tokens(midi)
        if apply_bpe_if_possible and self.has_bpe:
            self.apply_bpe(tokens)

        return tokens

    def complete_sequence(self, seq: TokSequence):
        r"""Completes (inplace) a :class:`miditok.TokSequence` object by converting its
        attributes. The input sequence can miss some of its attributes (ids, tokens),
        but needs at least one for reference. This method will create the missing ones
        from the present ones. The ``bytes`` attribute will be created if the tokenizer
        has been trained with BPE. The ``events`` attribute will not be filled as it is
        only intended for debug purpose.

        :param seq: input :class:`miditok.TokSequence`, must have at least one
            attribute defined.
        """
        if seq.tokens is None:
            if seq.events is not None:
                seq.tokens = self._events_to_tokens(seq.events)
            elif seq.ids is not None:
                seq.tokens = self._ids_to_tokens(seq.ids)
            elif seq.bytes is not None:
                seq.tokens = self._bytes_to_tokens(seq.bytes)
        if seq.ids is None:
            seq.ids = self._tokens_to_ids(seq.tokens)

        if self.has_bpe and seq.bytes is None:
            seq.bytes = self._ids_to_bytes(seq.ids, as_one_str=True)

    def _tokens_to_ids(
        self, tokens: Sequence[Union[str, List[str]]]
    ) -> List[Union[int, List[int]]]:
        r"""Converts a list of tokens (str) into their associated ids (int).

        :param tokens: list of tokens (str) to convert.
        :return: list of corresponding ids (int).
        """
        if len(tokens) == 0:
            return []
        if isinstance(tokens[0], (list, tuple)):
            ids = [
                [self.vocab[i][token] for i, token in enumerate(seq)] for seq in tokens
            ]
        else:
            ids = [self.vocab[token] for token in tokens]
        return ids

    def _ids_to_tokens(
        self, ids: List[Union[int, List[int]]], as_str: bool = True
    ) -> List[Union[Union[str, Event], List[Union[str, Event]]]]:
        r"""Converts a sequence of ids (int) to their associated tokens (str or Event).
        **This method will not work with ids encoded with BPE. You will need to decode
        them first (:py:meth:`miditok.MIDITokenizer.decode_bpe`).**

        :param ids: sequence of ids (int) to convert.
        :param as_str: return the tokens as string objects, otherwise Event objects
            (default: True)
        :return: the sequence of corresponding tokens (str or Event).
        """
        tokens = []
        if len(ids) == 0:
            return tokens
        if isinstance(ids[0], list):  # multiple vocabularies
            for (
                multi_ids
            ) in ids:  # cannot use recursion here because of the vocabulary type id
                multi_event = []
                for i, token in enumerate(multi_ids):
                    event_str = self[i, token]
                    multi_event.append(
                        event_str if as_str else Event(*event_str.split("_"))
                    )
                tokens.append(multi_event)
            return tokens

        for id_ in ids:
            event_str = self[id_]
            tokens.append(event_str if as_str else Event(*event_str.split("_")))
        return tokens

    @staticmethod
    def _events_to_tokens(
        events: List[Union[Event, List[Event]]],
    ) -> List[Union[str, List[str]]]:
        r"""Converts a sequence of Events to their associated tokens (str).

        :param events: sequence of Events to convert.
        :return: the sequence of corresponding tokens (str).
        """
        tokens = []
        if len(events) == 0:
            return tokens
        if isinstance(events[0], list):  # multiple vocabularies
            # cannot use recursion here because of the vocabulary type id
            return [[str(event) for event in multi_event] for multi_event in events]

        return [str(event) for event in events]

    def _ids_to_bytes(
        self, ids: List[Union[int, List[int]]], as_one_str: bool = False
    ) -> Union[str, List[str]]:
        r"""Converts a list of ids into their associated bytes.
        It can be returned either as a list of bytes or as a unique string of bytes.
        **This method will not work with ids encoded with BPE. You will need to decode
        them first (:py:meth:`miditok.MIDITokenizer.decode_bpe`).**

        :param ids: token ids (int) to convert.
        :param as_one_str: will return the bytes all concatenated into one string.
            (default: False)
        :return: the tokens converted into strings of unique bytes.
        """
        if len(ids) == 0:
            return ""
        if isinstance(ids[0], list):
            return [self._ids_to_bytes(item, as_one_str) for item in ids]
        bytes_ = [self._vocab_base_id_to_byte[i] for i in ids]
        return "".join(bytes_) if as_one_str else bytes_

    def _bytes_to_tokens(
        self, bytes_: Union[str, List[str]], as_str: bool = True
    ) -> List[Union[Union[str, Event], List[Union[str, Event]]]]:
        r"""Converts a sequence of bytes into their associated tokens (str or Event).

        :param bytes_: sequence of bytes to convert.
        :param as_str: return the events as string objects, otherwise Event objects
            (default: True)
        :return: the sequence of corresponding tokens (str).
        """
        if len(bytes_) == 0:
            return []
        if isinstance(bytes_[0], list):  # multiple vocabularies
            return [self._bytes_to_tokens(byte_) for byte_ in bytes_]

        tokens = []
        for byte_ in bytes_:
            token_str = self._vocab_bpe_bytes_to_tokens[byte_]
            tokens.append(token_str if as_str else Event(*token_str.split("_")))
        return [tok for toks in tokens for tok in toks]  # flatten

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        programs: Optional[List[Tuple[int, bool]]] = None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = None,
    ) -> Score:
        r"""Detokenize one or multiple sequences of tokens into a MIDI file.
        You can give the tokens sequences either as :class:`miditok.TokSequence`
        objects, lists of integers, numpy arrays or PyTorch / Tensorflow tensors.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence`, a Tensor (PyTorch and Tensorflow are
            supported), a numpy array or a Python list of ints. The first dimension
            represents tracks, unless the tokenizer handle tracks altogether as a
            single token sequence (``tokenizer.one_token_stream == True``).
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: None)
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the
            MIDI to create).
        :return: the midi object (symusic.Score).
        """
        midi = self._tokens_to_midi(tokens, programs, time_division)

        # Set default tempo and time signatures at tick 0 if not present
        if len(midi.tempos) == 0 or midi.tempos[0].time != 0:
            midi.tempos.insert(0, Tempo(0, self._DEFAULT_TEMPO))
        if len(midi.time_signatures) == 0 or midi.time_signatures[0].time != 0:
            midi.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump_midi(output_path)
        return midi

    def _tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        programs: Optional[List[Tuple[int, bool]]] = None,
        time_division: Optional[int] = None,
    ) -> Score:
        r"""Internal method called by ``self.tokens_to_midi``, intended to be
        implemented by inheriting classes.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence`, a Tensor (PyTorch and Tensorflow are
            supported), a numpy array or a Python list of ints. The first dimension
            represents tracks, unless the tokenizer handle tracks altogether as a
            single token sequence (``tokenizer.one_token_stream == True``).
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the
            MIDI to create).
        :return: the midi object (symusic.Score).
        """
        raise NotImplementedError

    @abstractmethod
    def _create_base_vocabulary(self, *args, **kwargs) -> List[Union[str, List[str]]]:
        r"""Creates the vocabulary, as a list of string tokens.
        This method is unimplemented and need to be overridden by inheriting classes.
        Each token as to be given as the form of "Type_Value", separated with an
        underscore. Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real"
        vocabulary as a dictionary. Do not include special tokens. These have to be
        given when creating the tokenizer, and will be added to the vocabulary by
        :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        raise NotImplementedError

    def __create_vocabulary(self):
        r"""Method actually creating the vocabulary object, as Dictionary, from the
        ``_create_vocabulary`` method implemented by tokenization classes.
        This method is called at ``__init__``\.
        """
        vocab = self._create_base_vocabulary()

        if isinstance(vocab[0], list):  # multi-voc
            self._vocab_base = [{} for _ in range(len(vocab))]
            self.__vocab_base_inv = [{} for _ in range(len(vocab))]
            for vid in range(len(vocab)):
                vocab[vid] = self.special_tokens + vocab[vid]
                for tok in vocab[vid]:
                    self.add_to_vocab(tok, vid)
        else:
            vocab = self.special_tokens + vocab
            for tok in vocab:
                self.add_to_vocab(tok)

    def _add_additional_tokens_to_vocab_list(self, vocab: List[str]):
        # PITCH INTERVALS
        if self.config.use_pitch_intervals:
            for interval_type in ("PitchIntervalTime", "PitchIntervalChord"):
                vocab += [
                    f"{interval_type}_{pitch}"
                    for pitch in range(
                        -self.config.max_pitch_interval,
                        self.config.max_pitch_interval + 1,
                    )
                ]

        # CHORD
        if self.config.use_chords:
            vocab += self._create_chords_tokens()

        # REST
        if self.config.use_rests:
            vocab += [f'Rest_{".".join(map(str, rest))}' for rest in self.rests]

        # TEMPO
        if self.config.use_tempos:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        # PROGRAM
        if self.config.use_programs:
            vocab += [f"Program_{program}" for program in self.config.programs]

        # TIME SIGNATURE
        if self.config.use_time_signatures:
            vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]

        # PEDAL
        if self.config.use_sustain_pedals:
            if self.config.use_programs:
                vocab += [f"Pedal_{program}" for program in self.config.programs]
                if not self.config.sustain_pedal_duration:
                    vocab += [f"PedalOff_{program}" for program in self.config.programs]
            else:
                vocab.append("Pedal_0")
                if not self.config.sustain_pedal_duration:
                    vocab.append("PedalOff_0")

        # PITCH BEND
        if self.config.use_pitch_bends:
            vocab += [f"PitchBend_{pitch_bend}" for pitch_bend in self.pitch_bends]

    def _update_token_types_indexes(self):
        r"""Updates the _token_types_indexes attribute according to _event_to_token."""

        def create_for_dict(voc: Dict[str, int]) -> Dict[str, List[int]]:
            types_ = {}
            for event, token in voc.items():
                token_type = event.split("_")[0]
                if token_type in types_:
                    types_[token_type].append(token)
                else:
                    types_[token_type] = [token]
            return types_

        if self.is_multi_voc:
            self._token_types_indexes = []
            for voc_i in self._vocab_base:
                self._token_types_indexes.append(create_for_dict(voc_i))
        else:
            self._token_types_indexes = create_for_dict(self._vocab_base)

    def token_ids_of_type(
        self, token_type: str, vocab_id: Optional[int] = None
    ) -> List[int]:
        r"""Returns the list of token ids of the given type.

        :param token_type: token type to get the associated token ids.
        :param vocab_id: index of the vocabulary associated to the token, if applicable.
            (default: None)
        :return: list of token ids.
        """
        try:
            return (
                self._token_types_indexes[token_type]
                if vocab_id is None
                else self._token_types_indexes[vocab_id][token_type]
            )
        except KeyError:  # no tokens of this type, returns an empty list
            return []

    def add_to_vocab(
        self,
        token: Union[str, Event],
        vocab_idx: Optional[int] = None,
        byte_: Optional[str] = None,
        add_to_bpe_model: bool = False,
    ):
        r"""Adds an event to the vocabulary. Its index (int) will be the length of the
        vocab.

        :param token: token to add, as a formatted string of the form "Type_Value",
            e.g. Pitch_80, or an Event.
        :param vocab_idx: idx of the vocabulary (in case of embedding pooling).
            (default: None)
        :param byte_: unique byte associated to the token. This is used when building
            the vocabulary with fast BPE. If None is given, it will default to
            ``chr(id_ + CHR_ID_START)`` . (default: None)
        :param add_to_bpe_model: the token will be added to the bpe_model vocabulary
            too. (default: None)
        """
        token_str = token if isinstance(token, str) else str(token)

        if vocab_idx is not None:
            self._vocab_base[vocab_idx][token_str] = len(self._vocab_base[vocab_idx])
            self.__vocab_base_inv[vocab_idx][
                len(self.__vocab_base_inv[vocab_idx])
            ] = token_str
        else:
            id_ = len(self._bpe_model.get_vocab()) if self.has_bpe else len(self.vocab)
            self._vocab_base[token_str] = id_
            self.__vocab_base_inv[len(self.__vocab_base_inv)] = token_str

            # For BPE
            if byte_ is None:
                byte_ = chr(id_ + CHR_ID_START)
            self._vocab_base_id_to_byte[
                id_
            ] = byte_  # these vocabs are created at init, when the
            self._vocab_base_byte_to_token[byte_] = token
            if self._bpe_model is not None and add_to_bpe_model:
                self._bpe_model.add_tokens([byte_])

    def _create_chords_tokens(self) -> List[str]:
        """Just create the *Chord* tokens that will populate the base vocabulary. This
        protected method is intended to be used by subclasses when creating their
        vocabularies.

        :return: chord tokens, created from the tokenizer's params
        """
        tokens = []
        if self.config.chord_tokens_with_root_note:
            tokens += [
                f"Chord_{root_note}:{chord_quality}"
                for chord_quality in self.config.chord_maps
                for root_note in PITCH_CLASSES
            ]
        else:
            tokens += [
                f"Chord_{chord_quality}" for chord_quality in self.config.chord_maps
            ]

        # Unknown chords
        if self.config.chord_unknown is not None:
            if self.config.chord_tokens_with_root_note:
                tokens += [
                    f"Chord_{root_note}:{UNKNOWN_CHORD_PREFIX}{i}"
                    for i in range(*self.config.chord_unknown)
                    for root_note in PITCH_CLASSES
                ]
            else:
                tokens += [f"Chord_{i}" for i in range(*self.config.chord_unknown)]

        return tokens

    def token_id_type(self, id_: int, vocab_id: Optional[int] = None) -> str:
        r"""Returns the type of the given token id.

        :param id_: token id to get the type.
        :param vocab_id: index of the vocabulary associated to the token, if
            applicable. (default: None)
        :return: the type of the token, as a string
        """
        token = self.__get_from_voc(id_, vocab_id)
        return token.split("_")[0]

    @abstractmethod
    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Creates a dictionary describing the possible token type successions.
        This method is unimplemented and need to be overridden by inheriting classes.
        See other classes (:class:`miditok.REMI._create_token_types_graph`, ...)
        for examples of how to implement it."""
        raise NotImplementedError

    def _add_special_tokens_to_types_graph(self):
        r"""Adds (inplace) special tokens types to the token types graph dictionary.
        Two exceptions are made for the special BOS (Beginning of Sequence) and EOS
        (End of Sequence) tokens: No token type can precede a BOS token, and EOS token
        cannot precede any other token.
        """
        original_token_types = list(self.tokens_types_graph.keys())
        for special_token in self.config.special_tokens:
            special_token_type = special_token.split("_")[0]
            if special_token_type == EOS_TOKEN_NAME:
                self.tokens_types_graph[EOS_TOKEN_NAME] = []
            else:
                self.tokens_types_graph[special_token_type] = (
                    original_token_types + list(self.config.special_tokens)
                )

            if special_token_type != BOS_TOKEN_NAME:
                for token_type in original_token_types:
                    self.tokens_types_graph[token_type].append(special_token_type)

    def __create_durations_tuples(self) -> List[Tuple[int, int, int]]:
        r"""Creates the possible durations in beat / position units, as tuple of the
        form: (beat, pos, res) where beat is the number of beats, pos the number of
        "samples" and res the beat resolution considered (samples per beat).
        Example: (2, 5, 8) means the duration is 2 beat long + position 5 / 8 of the
        ongoing beat In pure ticks we have:
        duration = (beat * res + pos) * time_division // res
        It is equivalent to: duration = nb_of_samples * ticks_per_sample
        So in the last example, if time_division is 384:
        duration = (2 * 8 + 5) * 384 // 8 = 1008 ticks

        :return: the duration bins.
        """
        durations = []
        for beat_range, beat_res in self.config.beat_res.items():
            durations += [
                (beat, pos, beat_res)
                for beat in range(*beat_range)
                for pos in range(beat_res)
            ]
        durations += [
            (
                max(max(self.config.beat_res)),
                0,
                self.config.beat_res[max(self.config.beat_res)],
            )
        ]  # the last one
        del durations[0]  # removes duration of 0
        return durations

    @staticmethod
    def _token_duration_to_ticks(
        token_duration: Union[str, Tuple[int, int, int]], time_division: int
    ) -> int:
        r"""Converts a *Duration* token value of the form x.x.x, for
        beat.position.resolution, in ticks. Can also be used for
        *TimeShift* tokens.

        :param token_duration: Duration / TimeShift token value.
        :param time_division: time division.
        :return: the duration / time-shift in ticks.
        """
        if isinstance(token_duration, str):
            token_duration = tuple(map(int, token_duration.split(".")))
        beat, pos, res = token_duration
        return (beat * res + pos) * time_division // res

    def _ticks_to_duration_tokens(
        self,
        duration: int,
        time_division: Optional[int],
        rest: bool = False,
    ) -> Tuple[List[Tuple[int, int, int]], List[int]]:
        r"""Converts a duration in ticks into a sequence of
        `Duration`/`TimeShift` values.

        :param duration: duration in tick to convert.
        :param time_division: time division of the MIDI being parsed. If none is given,
            the method will use `self._current_midi_metadata["time_division"]`.
            (default: None)
        :param rest: the duration is a rest, hence the created tokens will be based on
            the `self.rests` values.
        :return: list of associated token values, and the list of the elapsed offset in
            tick for each of these values.
        """
        if rest:
            dur_bins = self._rests_ticks
            dur_vals = self.rests
        else:
            dur_bins = self._durations_ticks
            dur_vals = self.durations
        min_dur = dur_bins[0]

        offset_times = [0]
        values = []
        while duration >= min_dur:
            if rest:
                index = np.where(dur_bins - duration <= 0)[0][-1]
            else:
                index = np.argmin(np.abs(dur_bins - duration))
            val = dur_vals[index]
            values.append(val)
            val_ticks = self._token_duration_to_ticks(val, time_division)
            duration -= val_ticks
            offset_times.append(val_ticks)
        del offset_times[0]

        return values, offset_times

    def __create_rests(self) -> List[Tuple[int, int, int]]:
        r"""Creates the possible rests in beat / position units, as tuple of the form:
        (beat, pos, res) where beat is the number of beats, pos the number of "samples"
        and res the beat resolution considered (samples per beat).
        It follows the same data representation than duration and time shifts.

        :return: the rests.
        """
        rests = []
        for beat_range, beat_res in self.config.beat_res_rest.items():
            rests += [
                (beat, pos, beat_res)
                for beat in range(*beat_range)
                for pos in range(beat_res)
            ]
        rests += [
            (
                max(max(self.config.beat_res_rest)),
                0,
                self.config.beat_res_rest[max(self.config.beat_res_rest)],
            )
        ]  # the last one
        del rests[0]  # removes rests of 0
        return rests

    def __create_tempos(self) -> np.ndarray:
        r"""Creates the possible tempos, as a float number array.

        The self.config.nb_tempos tempos are distributed in the self.config.tempo_range
        using either log or linear scaled values based on the value of
        ``self.config.log_tempos``.

        :return: the tempos.
        """
        tempo_fn = np.geomspace if self.config.log_tempos else np.linspace
        return tempo_fn(*self.config.tempo_range, self.config.num_tempos).round(2)

    def __create_time_signatures(self) -> List[Tuple]:
        r"""Creates the possible time signatures, as tuples of the form:
        (nb_beats, beat_res) where nb_beats is the number of beats per bar.
        Example: (3, 4) means one bar is 3 beat long and each beat is a quarter note.

        :return: the time signatures.
        """
        time_signature_range = self.config.time_signature_range

        time_signatures = []
        for beat_res, beats in time_signature_range.items():
            if beat_res <= 0 or not math.log2(beat_res).is_integer():
                raise ValueError(
                    f"The beat resolution ({beat_res}) in time signature must be a"
                    f"power of 2."
                )

            time_signatures.extend([(nb_beats, beat_res) for nb_beats in beats])

        return time_signatures

    def __create_pitch_bends(self) -> np.ndarray:
        r"""Creates the pitch bend values, as numpy array, using
        ``self.config.pitch_bend_range``.

        :return: the pitch bend values.
        """
        return np.linspace(*self.config.pitch_bend_range, dtype=np.int32)

    @staticmethod
    def _compute_ticks_per_bar(time_sig: TimeSignature, time_division: int):
        r"""Computes time resolution of one bar in ticks.

        :param time_sig: time signature object
        :param time_division: MIDI time division / resolution, in ticks/beat (of the
            MIDI being parsed)
        :return: MIDI bar resolution, in ticks/bar
        """
        return int(time_division * 4 * time_sig.numerator / time_sig.denominator)

    @staticmethod
    def _parse_token_time_signature(token_time_sig: str) -> Tuple[int, int]:
        r"""Converts a time signature token value of the form x/x into a tuple of
        integers, time signature's numerator (bar length in beats) and denominator
        (beat resolution).

        :param token_time_sig: TimeSig token value.
        :return: the numerator and denominator of a time signature.
        """
        numerator, denominator = map(int, token_time_sig.split("/"))
        return numerator, denominator

    def validate_midi_time_signatures(self, midi: Score) -> bool:
        r"""Checks if a MIDI contains only time signatures supported by the tokenizer.

        :param midi: MIDI file
        :return: boolean indicating whether the MIDI can be processed by the tokenizer.
        """
        if self.config.use_time_signatures:
            for time_sig in midi.time_signatures:
                if (
                    time_sig.numerator,
                    time_sig.denominator,
                ) not in self.time_signatures:
                    return False
        return True

    def learn_bpe(
        self,
        vocab_size: int,
        iterator: Optional[Iterable] = None,
        files_paths: Optional[List[Union[Path, str]]] = None,
        start_from_empty_voc: bool = False,
        **kwargs,
    ):
        r"""Method to construct the vocabulary from BPE, backed by the 🤗tokenizers
        library. The data used for training can either be given through the
        ``iterator`` argument as an iterable object yielding strings, or by
        ``tokens_paths`` as a list of paths to token json files that will be loaded.
        You can read the Hugging Face `🤗tokenizers documentation
        <https://huggingface.co/docs/tokenizers/training_from_memory>`_,
        `🤗tokenizers API documentation <https://huggingface.co/docs/tokenizers/python/v0.9.4/api/reference.html#>`_
        and `🤗tokenizers course <https://huggingface.co/course/chapter6/2?fw=pt>`_
        for more details about the ``iterator`` and input type.

        **The training progress bar will not appear with non-proper terminals.**
        (cf `GitHub issue <https://github.com/huggingface/tokenizers/issues/157>`_ )

        :param vocab_size: size of the vocabulary to learn / build.
        :param iterator: an iterable object yielding the training data, as lists of
            string. It can be a list or a Generator. This iterator will be passed to
            the BPE model for training. It musts implement the ``__len__`` method. If
            None is given, you must use the ``tokens_paths`` argument. (default: None)
        :param files_paths: paths of the files to load and use. They can be either MIDI
            or tokens (json) files. (default: None)
        :param start_from_empty_voc: the training will start from an empty base
            vocabulary. The tokenizer will then have a base vocabulary only based on
            the unique bytes present in the training data. If you set this argument to
            True, you should use the tokenizer only with the training data, as new data
            might contain "unknown" tokens missing from the vocabulary. Comparing this
            to text, setting this argument to True would create a tokenizer that will
            only know the characters present in the training data, and would not be
            compatible/know other characters. This argument can allow to optimize the
            vocabulary size. If you are unsure about this, leave it to False.
            (default: False)
        :param kwargs: any additional argument to pass to the trainer.
        """
        if self.is_multi_voc:
            warnings.warn(
                "This tokenizer is based on multiple vocabularies/embedding pooling."
                "It is therefore not compatible with Byte Pair Encoding (BPE). Skipping"
                "this method call (learn_bpe).",
                stacklevel=2,
            )
            return
        if iterator is None and files_paths is None:
            raise ValueError(
                "You must give an iterator or a list of paths to tokens to train the"
                "tokenizer with BPE."
            )

        if vocab_size <= len(self.vocab):
            warnings.warn(
                f"vocab_size ({vocab_size}) need to be higher than the size of the"
                f"current vocabulary ({len(self.vocab)}). Skipping BPE training.",
                stacklevel=2,
            )
            return

        # If no iterator, loads tokens / samples to analyze
        if iterator is None:
            iterator = BPEIterator(self, files_paths)

        # Create new tokenizer model
        if self._bpe_model is None or start_from_empty_voc:
            nb_bytes = (
                len(self.config.special_tokens)
                if start_from_empty_voc
                else len(self._vocab_base)
            )
            voc_start = {chr(i + CHR_ID_START): i for i in range(nb_bytes)}
            self._bpe_model = TokenizerFast(
                BPE(
                    vocab=voc_start,
                    merges=[],
                    dropout=None,
                    continuing_subword_prefix="",
                    end_of_word_suffix="",
                    fuse_unk=False,
                )
            )

        # Trains the tokenizer
        special_tokens_bytes = []
        if len(self.config.special_tokens) > 0:
            special_tokens_bytes = self._ids_to_bytes(
                self._tokens_to_ids(self.config.special_tokens)
            )
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens_bytes,
            show_progress=True,
            **kwargs,
        )
        self._bpe_model.train_from_iterator(
            iterator, length=len(iterator), trainer=trainer
        )

        # Update other vocabs accordingly
        if start_from_empty_voc:
            # If we do not give an existing vocabulary to the tokenizer, 🤗tokenizers
            # first fill its vocabulary with all bytes present in the training samples,
            # sorted by byte / char index. Some bytes / tokens might be missing from
            # tokenizer.get_vocab(), as simply not present in training samples. We must
            # get rid of them from the base vocabulary
            new_vocab = {
                k: v
                for k, v in sorted(
                    self._bpe_model.get_vocab().items(), key=lambda item: item[1]
                )
            }
            byte_to_token_old = deepcopy(self._vocab_base_byte_to_token)

            # Rebuild base vocabularies dicts
            self._vocab_base = {}  # token -> id
            self.__vocab_base_inv = {}  # id -> token
            self._vocab_base_byte_to_token = {}  # for all basic tokens
            self._vocab_base_id_to_byte = {}
            # dict is ordered so id val is incremented each time, from 0
            for byte_ in new_vocab:
                if byte_ in byte_to_token_old:
                    token = byte_to_token_old[
                        byte_
                    ]  # get the original token associated to the byte
                    self.add_to_vocab(
                        token, byte_=byte_, add_to_bpe_model=False
                    )  # adds it to _vocab_base

        # Update __vocab_bpe_bytes_to_tokens for faster decoding
        self._vocab_bpe_bytes_to_tokens = {
            k: [self._vocab_base_byte_to_token[b] for b in k]
            for k in self._bpe_model.get_vocab()
        }

        self.has_bpe = True

    def apply_bpe(self, seq: Union[TokSequence, List[TokSequence]]):
        """Applies Byte Pair Encoding (BPE) to a TokSequence, or list of TokSequences.
        If a list is given, BPE will be applied by batch on all sequences at the time.

        :param seq: Sequence(s) to apply BPE.
        """
        if isinstance(seq, list):
            for seq_ in seq:
                self.complete_sequence(seq_)
            encoded_tokens = self._bpe_model.encode_batch(
                [[t.bytes] for t in seq], is_pretokenized=True
            )
            for seq_, bpe_tokens in zip(seq, encoded_tokens):
                seq_.ids = bpe_tokens.ids
                seq_.ids_bpe_encoded = True

        else:
            self.complete_sequence(seq)
            encoded_tokens = self._bpe_model.encode([seq.bytes], is_pretokenized=True)
            seq.ids = encoded_tokens.ids
            seq.ids_bpe_encoded = True

    def _are_ids_bpe_encoded(self, ids: Union[List[int], np.ndarray]) -> bool:
        r"""A small check telling if a sequence of ids are encoded with BPE.
        This is performed by checking if any id has a value superior or equal to the
        length of the base vocabulary.

        :param ids: ids to check
        :return: boolean, True if ids are encoded with BPE, False otherwise.
        """
        return np.any(np.array(ids) >= len(self._vocab_base))

    def decode_bpe(self, seq: Union[TokSequence, List[TokSequence]]):
        r"""Decodes (inplace) a sequence of tokens (:class:`miditok.TokSequence`) with
        ids encoded with BPE. This method only modifies the ``.ids`` attribute of the
        input sequence(s) only and does not complete it. This method can also receive a
        list of sequences, in which case it will decompose BPE on each of them
        recursively.

        :param seq: token sequence to decompose.
        """

        if isinstance(seq, list):
            [self.decode_bpe(seq_) for seq_ in seq]

        elif isinstance(seq, TokSequence) and seq.ids_bpe_encoded:
            encoded_bytes = [self._bpe_model.id_to_token(id_) for id_ in seq.ids]
            decoded_tokens = [
                self._vocab_bpe_bytes_to_tokens[byte_] for byte_ in encoded_bytes
            ]
            decoded_tokens = [
                item for sublist in decoded_tokens for item in sublist
            ]  # flatten
            seq.tokens = decoded_tokens
            seq.ids = self._tokens_to_ids(decoded_tokens)
            seq.ids_bpe_encoded = False

    def tokenize_midi_dataset(
        self,
        midi_paths: Union[str, Path, Sequence[Union[str, Path]]],
        out_dir: Union[str, Path],
        overwrite_mode: bool = True,
        validation_fn: Optional[Callable[[Score], bool]] = None,
        data_augment_offsets=None,
        save_programs: Optional[bool] = None,
        logging: bool = True,
    ):
        r"""Converts a dataset / list of MIDI files, into their token version and save
        them as json files. The resulting json files will have an "ids" entry containing
        the token ids. The format of the ids will correspond to the format of the
        tokenizer (``tokenizer.io_format``). Note that the file tree of the source
        files, up to the deepest common root directory if `midi_paths` is given as a
        list of paths, will be reproducing in ``out_dir``. The config of the tokenizer
        will be saved as a file named ``tokenizer_config_file_name`` (default:
        ``tokenizer.conf``) in the ``out_dir`` directory.

        :param midi_paths: paths of the MIDI files. It can also be a path to a
            directory, in which case this method will recursively find the MIDI files
            within (.mid, .midi extensions).
        :param out_dir: output directory to save the converted files.
        :param overwrite_mode: if True, will overwrite files if they already exist when
            trying to save the new ones created by the method. This is enabled by
            default, as it is good practice to use dedicated directories for each
            tokenized dataset. If False, if a file already exist, the new one will be
            saved in the same directory, with the same name with a number appended at
            the end. Both token files and tokenizer config are concerned.
            (default: True)
        :param validation_fn: a function checking if the MIDI is valid on your
            requirements (e.g. time signature, minimum/maximum length, instruments...).
        :param data_augment_offsets: data augmentation arguments, to be passed to the
            miditok.data_augmentation.data_augmentation_dataset method. Has to be given
            as a list / tuple of offsets pitch octaves, velocities, durations, and
            finally their directions (up/down). (default: None)
        :param save_programs: will save the programs of the tracks of the MIDI as an
            entry in the Json file. That this option is probably unnecessary when using
            a multitrack tokenizer (`config.use_programs`), as the program information
            is present within the tokens, and that the tracks having the same programs
            are likely to have been merged. (default: False if ``config.use_programs``,
            else True)
        :param logging: logs progress bar.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # User gave a path to a directory, we'll scan it to find MIDI files
        if not isinstance(midi_paths, Sequence):
            if isinstance(midi_paths, str):
                midi_paths = Path(midi_paths)
            root_dir = midi_paths
            midi_paths = sum(
                (list(midi_paths.glob(f"**/*{ext}")) for ext in MIDI_FILES_EXTENSIONS),
                [],
            )
        # User gave a list of paths, we need to find the root / deepest common subdir
        else:
            all_parts = [Path(path).parent.parts for path in midi_paths]
            max_depth = max(len(parts) for parts in all_parts)
            root_parts = []
            for depth in range(max_depth):
                if len(set(parts[depth] for parts in all_parts)) > 1:
                    break
                root_parts.append(all_parts[0][depth])
            root_dir = Path(*root_parts)

        if save_programs is None:
            save_programs = not self.config.use_programs

        desc = f'Tokenizing MIDIs ({"/".join(list(out_dir.parts[-2:]))})'
        for midi_path in tqdm(midi_paths, desc=desc):
            # Some MIDIs can contain errors, if so the loop continues
            midi_path = Path(midi_path)
            try:
                midi = Score(midi_path)
            except FileNotFoundError:
                if logging:
                    warnings.warn(f"File not found: {midi_path}", stacklevel=2)
                continue
            except MIDI_LOADING_EXCEPTION:
                continue

            # Checks the time division is valid
            if midi.ticks_per_quarter < max(self.config.beat_res.values()) * 4:
                continue
            # Passing the MIDI to validation tests if given
            if validation_fn is not None and not validation_fn(midi):
                continue

            # Checks if MIDI contains supported time signatures
            if not self.validate_midi_time_signatures(midi):
                continue

            # Tokenizing the MIDI, without BPE here as this will be done at the end as
            # we might perform data aug before
            tokens = self(midi, apply_bpe_if_possible=False)

            # Data augmentation on tokens
            if data_augment_offsets is not None:
                if isinstance(tokens, TokSequence):
                    tokens = [tokens]
                offsets = get_offsets(
                    self,
                    *data_augment_offsets,
                    ids=[seq.ids for seq in tokens],
                )
                corrected_offsets = deepcopy(offsets)
                vel_dim = int(128 / len(self.velocities))
                corrected_offsets[1] = [
                    int(off / vel_dim) for off in corrected_offsets[1]
                ]

                augmented_tokens: Dict[
                    Tuple[int, int, int], Union[TokSequence, List[TokSequence]]
                ] = {}
                for track_seq, is_drum in zip(
                    tokens, [track.is_drum for track in midi.tracks]
                ):
                    if is_drum:
                        continue
                    aug = data_augmentation_tokens(
                        track_seq.ids,
                        self,
                        *corrected_offsets,
                        need_to_decode_bpe=False,
                    )
                    if len(aug) == 0:
                        continue
                    for aug_offsets, aug_ids in aug:
                        seq = TokSequence(ids=aug_ids)
                        if self.one_token_stream:
                            augmented_tokens[aug_offsets] = seq
                            continue
                        try:
                            augmented_tokens[aug_offsets].append(seq)
                        except KeyError:
                            augmented_tokens[aug_offsets] = [seq]

                if not self.one_token_stream:
                    for i, (seq, is_drum) in enumerate(
                        zip(tokens, [track.is_drum for track in midi.tracks])
                    ):  # adding drums to all already augmented
                        if is_drum:
                            for aug_offsets in augmented_tokens:
                                augmented_tokens[aug_offsets].insert(
                                    i, TokSequence(ids=seq.ids)
                                )

                tokens = [((0, 0, 0), tokens)]
                tokens += [(offs, seqs) for offs, seqs in augmented_tokens.items()]
            else:
                tokens = [((0, 0, 0), tokens)]

            # Apply BPE on tokens
            if self.has_bpe:
                if self.one_token_stream:
                    self.apply_bpe([seq for _, seq in tokens])
                else:
                    for _, track_seqs in tokens:
                        self.apply_bpe(track_seqs)

            # Set output file path
            tokens_dir = out_dir / midi_path.parent.relative_to(root_dir)
            tokens_dir.mkdir(parents=True, exist_ok=True)

            # Save tokens files
            for aug_offsets, seq in tokens:
                suffix = ""
                if any(off != 0 for off in aug_offsets):
                    suffix = "§" + "_".join(
                        [
                            f"{t}{offset}"
                            for t, offset in zip(["p", "v", "d"], aug_offsets)
                            if offset != 0
                        ]
                    )
                out_path = tokens_dir / f"{midi_path.stem}{suffix}.json"
                if not overwrite_mode and out_path.is_file():
                    i = 1
                    while out_path.is_file():
                        out_path = out_path.parent / f"{midi_path.stem}_{i}.json"
                        i += 1

                # Save the tokens as JSON
                self.save_tokens(
                    seq,
                    out_path,
                    get_midi_programs(midi) if save_programs else None,
                )

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(
        self, tokens: Union[TokSequence, List[Union[int, List[int]]]]
    ) -> Union[float, List[float]]:
        r"""Checks if a sequence of tokens is made of good token types successions and
        returns the error ratio (lower is better). The common implementation in
        MIDITokenizer class will check token types, duplicated notes and time errors.
        It works for ``REMI``, ``TSD`` and ``Structured``. Other tokenizations override
        this method to include other errors (like no *NoteOff* / *NoteOn* for
        ``MIDILike`` and embedding pooling). Overridden methods must call
        ``decompose_bpe`` at the beginning if BPE is used.

        :param tokens: sequence of tokens to check.
        :return: the error ratio (lower is better).
        """
        # If list of TokSequence -> recursive
        if isinstance(tokens, list):
            return [self.tokens_errors(tok_seq) for tok_seq in tokens]
        elif len(tokens) == 0:
            return 0

        nb_tok_predicted = len(tokens)  # used to norm the score
        if self.has_bpe:
            self.decode_bpe(tokens)
        self.complete_sequence(tokens)

        # Override from here
        tokens = tokens.tokens

        err_type = 0  # i.e. incompatible next type predicted
        err_time = 0  # i.e. goes back or stay in time (does not go forward)
        err_note = 0  # i.e. duplicated
        previous_type = tokens[0].split("_")[0]
        current_pos = -1
        current_program = 0
        current_pitches = {p: [] for p in self.config.programs}
        previous_pitch_onset = {program: -128 for program in self.config.programs}
        previous_pitch_chord = {program: -128 for program in self.config.programs}
        note_tokens_types = ["Pitch", "NoteOn"]
        if self.config.use_pitch_intervals:
            note_tokens_types += ["PitchIntervalTime", "PitchIntervalChord"]

        # Init first note and current pitches if needed
        if previous_type in note_tokens_types:
            pitch_val = int(tokens[0].split("_")[1])
            current_pitches[current_program].append(pitch_val)
        elif previous_type == "Position":
            current_pos = int(tokens[0].split("_")[1])

        for ti, token in enumerate(tokens[1:]):
            # err_tokens = tokens[ti - 4 : ti + 4]  # uncomment for debug
            event_type, event_value = token.split("_")

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type == "Bar":  # reset
                    current_pos = -1
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in ["TimeShift", "Time-Shift", "Rest"]:
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in note_tokens_types:
                    if event_type == "Pitch":
                        pitch_val = int(event_value)
                        previous_pitch_onset[current_program] = pitch_val
                        previous_pitch_chord[current_program] = pitch_val
                    elif event_type == "PitchIntervalTime":
                        pitch_val = previous_pitch_onset[current_program] + int(
                            event_value
                        )
                        previous_pitch_onset[current_program] = pitch_val
                        previous_pitch_chord[current_program] = pitch_val
                    else:  # PitchIntervalChord
                        pitch_val = previous_pitch_chord[current_program] + int(
                            event_value
                        )
                        previous_pitch_chord[current_program] = pitch_val
                    if pitch_val in current_pitches[current_program]:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches[current_program].append(pitch_val)
                elif event_type == "Position":
                    # With time signatures, it can happen that Rest -> TimeSig ->
                    # Position
                    if (
                        int(event_value) <= current_pos
                        and previous_type != "Rest"
                        and not (
                            previous_type == "TimeSig"
                            and tokens[ti - 1].split("_")[0] == "Rest"
                        )
                    ):
                        err_time += 1  # token position value <= to the current pos
                    current_pos = int(event_value)
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type == "Program":  # reset
                    current_program = int(event_value)
            # Bad token type
            else:
                err_type += 1
            previous_type = event_type

        return (err_type + err_time + err_note) / nb_tok_predicted

    def save_tokens(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        path: Union[str, Path],
        programs: Optional[List[Tuple[int, bool]]] = None,
        **kwargs,
    ):
        r"""Saves tokens as a JSON file.
        In order to reduce disk space usage, **only the ids are saved**. Use kwargs to
        save any additional information within the JSON file.

        :param tokens: tokens, as list, numpy array, torch or tensorflow Tensor.
        :param path: path of the file to save.
        :param programs: (optional), programs of the associated tokens, should be given
            as a tuples (int, bool) for (program, is_drum).
        :param kwargs: any additional information to save within the JSON file.
        """
        ids = []
        ids_bpe_encoded = None

        if isinstance(tokens, TokSequence):
            self.complete_sequence(tokens)
            ids_bpe_encoded = tokens.ids_bpe_encoded
            ids = tokens.ids
        elif isinstance(tokens, list) and len(tokens) == 0:
            pass
        elif isinstance(tokens[0], TokSequence):
            ids_bpe_encoded = []
            for seq in tokens:
                self.complete_sequence(seq)
                ids_bpe_encoded.append(seq.ids_bpe_encoded)
                ids.append(seq.ids)
        else:
            ids = convert_ids_tensors_to_list(tokens)

        if "ids_bpe_encoded" not in kwargs and ids_bpe_encoded is not None:
            kwargs["ids_bpe_encoded"] = ids_bpe_encoded

        with Path(path).open("w") as outfile:
            dic = {"ids": ids, **kwargs}
            if programs is not None:
                dic["programs"] = programs
            json.dump(dic, outfile)

    @staticmethod
    def load_tokens(path: Union[str, Path]) -> Union[List[Any], Dict]:
        r"""Loads tokens saved as JSON files.

        :param path: path of the file to load.
        :return: the tokens, with the associated information saved with.
        """
        with Path(path).open() as file:
            return json.load(file)

    def _save_pretrained(self, *args, **kwargs):
        # called by `ModelHubMixin.from_pretrained`.
        return self.save_params(*args, **kwargs)

    def save_params(
        self,
        out_path: Union[str, Path],
        additional_attributes: Optional[Dict] = None,
        filename: Optional[str] = DEFAULT_TOKENIZER_FILE_NAME,
    ):
        r"""Saves the config / parameters of the tokenizer in a json encoded file. This
        can be useful to keep track of how a dataset has been tokenized.
        **Note:** if you override this method, you should probably call it (super()) at
        the end and use the additional_attributes argument.

        :param out_path: output path to save the file. This can be either a path to a
            file (with a name and extension), or a path to a directory in which case
            the ``filename`` argument will be used.
        :param additional_attributes: any additional information to store in the config
            file. It can be used to override the default attributes saved in the parent
            method. (default: None)
        :param filename: name of the file to save, to be used in case `out_path` leads
            to a directory. (default: ``"tokenizer.conf"``)
        """
        if additional_attributes is None:
            additional_attributes = {}
        if self.has_bpe:  # saves whole vocab if BPE
            additional_attributes["_vocab_base"] = self._vocab_base
            additional_attributes["_bpe_model"] = self._bpe_model.to_str()
            additional_attributes[
                "_vocab_base_byte_to_token"
            ] = self._vocab_base_byte_to_token

        dict_config = self.config.to_dict(serialize=True)
        for beat_res_key in ["beat_res", "beat_res_rest"]:
            dict_config[beat_res_key] = {
                f"{k1}_{k2}": v for (k1, k2), v in dict_config[beat_res_key].items()
            }
        params = {
            "config": dict_config,
            "one_token_stream": self.one_token_stream,
            "has_bpe": self.has_bpe,
            "tokenization": self.__class__.__name__,
            "miditok_version": CURRENT_MIDITOK_VERSION,
            "symusic_version": CURRENT_SYMUSIC_VERSION,
            "hf_tokenizers_version": CURRENT_TOKENIZERS_VERSION,
            **additional_attributes,
        }

        out_path = Path(out_path)
        if out_path.is_dir() or "." not in out_path.name:
            out_path /= filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as outfile:
            json.dump(params, outfile, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **kwargs,
    ) -> "MIDITokenizer":
        # Called by `ModelHubMixin.from_pretrained`
        pretrained_path = Path(model_id)
        if pretrained_path.is_file():
            params_path = pretrained_path
        else:
            filename = kwargs.get("filename", DEFAULT_TOKENIZER_FILE_NAME)
            if (pretrained_path / filename).is_file():
                params_path = pretrained_path / filename
            else:
                params_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    library_name="MidiTok",
                    library_version=CURRENT_MIDITOK_VERSION,
                )

        return cls(params=params_path)

    def _load_params(self, config_file_path: Union[str, Path]):
        r"""Loads the parameters of the tokenizer from a config file. This method is
        not intended to be called outside __init__, when creating a tokenizer.

        :param config_file_path: path to the tokenizer config file (encoded as json).
        """
        with Path(config_file_path).open() as param_file:
            params = json.load(param_file)

        # Grab config, or creates one with default parameters (for retro-compatibility
        # with previous version)
        self.config = TokenizerConfig()
        config_attributes = list(self.config.to_dict().keys())
        old_add_tokens_attr = {
            "Chord": "use_chords",
            "Rest": "use_rests",
            "Tempo": "use_tempos",
            "TimeSignature": "use_time_signatures",
            "Program": "use_program",
        }

        # Overwrite config attributes
        for key, value in params.items():
            if key in ["tokenization", "miditok_version"]:
                continue
            elif key == "_vocab_base":
                self._vocab_base = value
                self.__vocab_base_inv = {v: k for k, v in value.items()}
                continue
            elif key == "_bpe_model":
                # using 🤗tokenizers builtin method
                self._bpe_model = TokenizerFast.from_str(value)
                continue
            elif key == "_vocab_base_byte_to_token":
                self._vocab_base_byte_to_token = value
                token_to_byte = {v: k for k, v in value.items()}
                self._vocab_base_id_to_byte = {
                    i: token_to_byte[tok] for tok, i in self._vocab_base.items()
                }
                self._vocab_bpe_bytes_to_tokens = {
                    k: [self._vocab_base_byte_to_token[b] for b in k]
                    for k in self._bpe_model.get_vocab()
                }
                continue
            elif key == "config":
                for beat_res_key in ["beat_res", "beat_res_rest"]:
                    # check here for previous versions (< v2.1.5)
                    if beat_res_key in value:
                        value[beat_res_key] = {
                            tuple(map(int, beat_range.split("_"))): res
                            for beat_range, res in value[beat_res_key].items()
                        }
                value["time_signature_range"] = {
                    int(res): beat_range
                    for res, beat_range in value["time_signature_range"].items()
                }
                # Rest param < v2.1.5
                if "rest_range" in value:
                    value["rest_range"] = {
                        (0, value["rest_range"][1]): value["rest_range"][0]
                    }
                value = TokenizerConfig.from_dict(value)
            elif key in config_attributes:
                if key == "beat_res":
                    value = {
                        tuple(map(int, beat_range.split("_"))): res
                        for beat_range, res in value.items()
                    }
                elif key == "time_signature_range":
                    value = {int(res): beat_range for res, beat_range in value.items()}
                # Convert old attribute from < v2.1.0 to new for TokenizerConfig
                elif key in old_add_tokens_attr:
                    key = old_add_tokens_attr[key]
                setattr(self.config, key, value)
                continue
            elif key == "unique_track":
                # For config files <= v2.1.1 before the attribute is renamed
                self.one_token_stream = value

            setattr(self, key, value)

    @property
    def is_multi_voc(self) -> bool:
        """Returns a bool indicating if the tokenizer uses embedding
        pooling, and so have multiple vocabularies.

        :return: True is the tokenizer uses embedding pooling else False.
        """
        return isinstance(self._vocab_base, list)

    @property
    def io_format(self) -> Tuple[str]:
        format_ = []
        if not self.one_token_stream:
            format_.append("I")
        format_.append("T")
        if self.is_multi_voc:
            format_.append("C")

        return tuple(d for d in format_)

    def __call__(self, obj: Any, *args, **kwargs):
        r"""Calling a tokenizer allows to directly convert a MIDI to tokens or the
        other way around. The method automatically detects MIDI and token objects, as
        well as paths and can directly load MIDI or token json files before converting
        them. This will call the :py:func:`miditok.MIDITokenizer.midi_to_tokens` if you
        provide a MIDI object or path to a MIDI file, or the
        :py:func:`miditok.MIDITokenizer.tokens_to_midi` method otherwise.

        :param obj: a `symusic.Score` object, a sequence of tokens, or a path to
            a MIDI or tokens json file.
        :return: the converted object.
        """
        # Tokenize MIDI
        if isinstance(obj, ScoreTick):
            return self.midi_to_tokens(obj, *args, **kwargs)

        # Loads a file (.mid or .json)
        elif isinstance(obj, (str, Path)):
            path = Path(obj)
            if path.suffix in MIDI_FILES_EXTENSIONS:
                midi = Score(obj)
                return self.midi_to_tokens(midi, *args, **kwargs)
            else:
                tokens = self.load_tokens(path)
                return self.tokens_to_midi(tokens, *args, **kwargs)

        # Depreciated miditoolkit object
        elif MidiFile is not None and isinstance(obj, MidiFile):
            warnings.warn(
                "You are using a depreciated `miditoolkit.MidiFile` object. MidiTok"
                "is now (>v3.0.0) using symusic.Score as MIDI backend. Your MIDI will"
                "be converted on the fly, however please consider using symusic.",
                stacklevel=2,
            )
            return self.midi_to_tokens(miditoolkit_to_symusic(obj), *args, **kwargs)

        # Consider it tokens --> converts to MIDI
        else:
            return self.tokens_to_midi(obj, *args, **kwargs)

    def __len__(self) -> int:
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling/have multiple vocabularies, it will return the **sum** of their
        lengths. If the vocabulary was learned with fast BPE, it will return the
        length of the BPE vocabulary, i.e. the proper number of possible token ids.
        Otherwise, it will return the length of the base vocabulary. Use the
        :py:func:`miditok.MIDITokenizer.len` property (``tokenizer.len``) to have the
        list of lengths.

        :return: length of the vocabulary.
        """
        if self.is_multi_voc:
            return sum([len(v) for v in self.vocab])
        elif self.has_bpe:
            return len(self._bpe_model.get_vocab())
        return len(self.vocab)

    @property
    def len(self) -> Union[int, List[int]]:  # noqa: A003
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling/have multiple vocabularies, it will return the **list** of their
        lengths. Use the :py:func:`miditok.MIDITokenizer.__len__` magic method
        (``len(tokenizer)``) to get the sum of the lengths.

        :return: length of the vocabulary.
        """
        return [len(v) for v in self.vocab] if self.is_multi_voc else len(self)

    def __repr__(self):
        out_str = f"{self.len} tokens with {self.io_format} io format"

        # one_token_stream / multi-voc
        tmp = []
        if self.one_token_stream:
            tmp.append("one token stream")
        if self.is_multi_voc:
            tmp.append("multi-voc")
        if len(tmp) > 0:
            out_str += f"({', '.join(tmp)})"

        # BPE
        if self.has_bpe:
            out_str += ", with BPE"
        else:
            out_str += ", without BPE"
        return out_str

    def __getitem__(
        self, item: Union[int, str, Tuple[int, Union[int, str]]]
    ) -> Union[str, int, List[int]]:
        r"""Convert a token (int) to an event (str), or vice-versa.

        :param item: a token (int) or an event (str). For tokenizers with
            embedding pooling/multiple vocabularies ( `tokenizer.is_multi_voc` ), you
            must either provide a string (token) that is within all vocabularies (e.g.
            special tokens), or a tuple where the first element in the index of the
            vocabulary and the second the element to index.
        :return: the converted object.
        """
        if isinstance(item, tuple) and self.is_multi_voc:
            return self.__get_from_voc(item[1], item[0])
        elif self.is_multi_voc and isinstance(item, str):
            if all(item in voc for voc in self.vocab):
                return [voc[item] for voc in self.vocab]
            else:
                raise ValueError(
                    "This tokenizer uses multiple vocabularies / embedding pooling. To"
                    "index it you must either provide a token (string) that is within"
                    "all vocabularies (e.g. special tokens), or a tuple where the"
                    "first element in the index of the vocabulary and the second the"
                    "element to index."
                )
        else:
            return self.__get_from_voc(item)

    def __get_from_voc(
        self, item: Union[int, str], vocab_id: Optional[int] = None
    ) -> Union[int, str]:
        r"""Get element from the vocabulary.
        The method handles both token (int) <--> event (str) ways.

        :param item: item to get / index.
        :param vocab_id: index of the vocabulary associated to the token, if
            applicable. (default: None)
        :return: the associated value.
        """
        if isinstance(item, str):
            voc = self.vocab[vocab_id] if self.is_multi_voc else self.vocab
        else:
            voc = (
                self.__vocab_base_inv[vocab_id]
                if self.is_multi_voc
                else self.__vocab_base_inv
            )
        return voc[item]

    def __eq__(self, other) -> bool:
        r"""Checks if two tokenizers are identical. This is done by comparing their
        vocabularies, and configuration.

        :param other: tokenizer to compare.
        :return: True if the vocabulary(ies) are identical, False otherwise.
        """
        if isinstance(other, MIDITokenizer):
            bpe_voc_eq = True
            if self._bpe_model is not None and other._bpe_model is not None:
                bpe_voc_eq = self._bpe_model.get_vocab() == other._bpe_model.get_vocab()
            return (
                self._vocab_base == other._vocab_base
                and bpe_voc_eq
                and self._vocab_base_byte_to_token == other._vocab_base_byte_to_token
                and self.config == other.config
            )
        return False
