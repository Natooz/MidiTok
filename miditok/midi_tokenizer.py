"""Base tokenizer class, acting as a "framework" for all tokenizers.
# TODO build docs action, make sure no error / warning https://github.com/readthedocs/actions.
"""
from __future__ import annotations

import json
import math
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from huggingface_hub import ModelHubMixin as HFHubMixin
from huggingface_hub import hf_hub_download
from symusic import (
    Score,
    Tempo,
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
from .utils import (
    compute_ticks_per_bar,
    compute_ticks_per_beat,
    convert_ids_tensors_to_list,
    detect_chords,
    get_midi_programs,
    get_midi_ticks_per_beat,
    merge_same_program_tracks,
    remove_duplicated_notes,
)
from .utils.utils import miditoolkit_to_symusic, np_get_closest


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
        params: str | Path | None = None,
    ) -> None:
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

        # Set one_token_stream mode according to the config params
        if self.config.use_programs:
            self.one_token_stream = self.config.one_token_stream_for_programs

        # Time Signatures
        # Need to be set before creating duration values/tokens.
        self.time_signatures = [TIME_SIGNATURE]
        if self.config.use_time_signatures:
            self.time_signatures = self.__create_time_signatures()

        # New time division to apply when preprocessing MIDIs
        # This is left as a class attribute and not a property as the config is not
        # intended to be modified after its creation. Ultimately, this could be
        # ensured by converting TokenizerConfig to a frozen dataclass.
        # The tokenizer's time division is chosen in order to make sure that is equal
        # to the lowest possible ticks per beat value, which depends on the supported
        # time signatures.
        # It shouldn't be used in place of the real ticks/beat value, which depends on
        # the current time signature denominator. The only exception is for tokenizers
        # which does not support time signature, i.e. which only consider 4/4.
        # TODO this doesn't work as is yet for non-beat-based tokenizers, as the
        #   `midi.resample` operation will keep too much accuracy for sections where
        #   the time signature is < 8, i.e. ticks per beat < to tokenizer.time_division
        #   We would need to resample per ticks/beat.
        #   Setting
        #   `self.time_division = max(res for res in self.config.beat_res.values())`
        #   makes the tests pass as no MIDI in the tests has a time signature of */8.
        #   https://github.com/Yikai-Liao/symusic/issues/10
        # TODO document this somewhere.
        #   Ideally the tokenizer's time division is set to the highest possible
        #   ticks per beat value, which depends on the highest `config.beat_res` given
        #   by the user and the maximum time signature denominator supported.
        #   This would allow to keep the maximum time information when tokenizing.
        #   In practice, if we do this, we would need to resample the time (onsets)
        #   of all the MIDI messages differently for every every portions having
        #   different ticks/beat values (can vary depending on the time signatures).
        #   This would add a significant additional preprocessing time if we do it in
        #   Python, so we do not do it. Instead we select the time division as the
        #   highest ticks/beat in `config.beat_res`, and round the duration values of
        #   the concerned tokens (note and time durations), even for
        #   *NoteOff*/*PedalOff* tokens (which is necessary).
        #   Ultimately, a resampling by ticks/beat could be implemented in a C++
        #   preprocessing step with pybind. We could then remove the `ceil` from the
        #   `MIDITokenizer_time_token_to_ticks` method.
        # self.time_division = max(res for res in self.config.beat_res.values())
        tpb_max_tokens = max(res for res in self.config.beat_res.values())
        denom_max = max(ts[1] for ts in self.time_signatures)
        quarter_factor = denom_max / 4  # can be < 1 if only */2 time sigs
        self.time_division = int(tpb_max_tokens * quarter_factor)

        # Durations
        # Usages:
        # Duration: tpb --> np.array (ticks) to get the closest;
        # Duration/TimeShift/Rest: ticks + tpb --> token (str);
        # Duration/TimeShift/Rest: token + tpb --> ticks (int);
        self.durations = self.__create_durations_tuples()
        self._tpb_to_time_array = self.__create_tpb_to_ticks_array()
        self._tpb_tokens_to_ticks = self.__create_tpb_tokens_to_ticks()
        self._tpb_ticks_to_tokens = self.__create_tpb_ticks_to_tokens()

        # Rests
        self.rests = []
        if self.config.use_rests:
            self.rests = self.__create_rests()
        self._tpb_to_rest_array = self.__create_tpb_to_ticks_array(rest=True)
        self._tpb_rests_to_ticks = self.__create_tpb_tokens_to_ticks(rest=True)

        # Velocities
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
        self.default_tempo = TEMPO
        if self.config.use_tempos:
            self.tempos = self.__create_tempos()
            self.default_tempo = self.tempos[np.argmin(np.abs(self.tempos - TEMPO))]

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

        # For logging
        self._verbose = False

    def _tweak_config_before_creating_voc(self) -> None:
        # called after setting the tokenizer's TokenizerConfig (.config). To be
        # customized by tokenizer classes.
        pass

    @property
    def vocab(
        self,
    ) -> dict[str, int] | list[dict[str, int]]:  # token (str) to its id (int)
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
    def vocab_bpe(self) -> None | dict[str, int]:  # byte (str) to its id (int)
        r"""Returns the vocabulary learnt with BPE.
        In case the tokenizer has not been trained with BPE, it returns None.

        :return: the BPE model's vocabulary.
        """
        if not self.has_bpe:
            return None
        else:
            return self._bpe_model.get_vocab()

    @property
    def special_tokens(self) -> list[str]:
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

    def _min_rest(self, ticks_per_beat: int) -> int:
        """Returns the minimum rest value in ticks, for a given ``ticks_per_beat``.

        :param ticks_per_beat: number of ticks in a beat. This depends on the current
            time signature, and is equal to the MIDI's time division if the denominator
            is 4 (quarter).
        :return: minimum rest in ticks.
        """
        if not self.config.use_rests:
            return 0
        else:
            return int(self._tpb_to_rest_array[ticks_per_beat][0])

    def preprocess_midi(self, midi: Score) -> Score:
        r"""Pre-process a MIDI file to resample its time and events values
        before tokenizing it. Its notes attributes (times, pitches, velocities) will be
        downsampled and sorted, duplicated notes removed, as well as tempos. Empty
        tracks (with no note) will be removed from the MIDI object. Notes with pitches
        outside ``self.config.pitch_range`` will be deleted.

        :param midi: MIDI object to preprocess.
        """
        # Resample time, not inplace
        # We do it even if the tokenizer's time division is superior to the MIDI's, as
        # it is the one used in all the downstream methods to compute the number of
        # ticks per bar/beat.
        if self.time_division != midi.ticks_per_quarter:
            midi = midi.resample(self.time_division, min_dur=1)

        # Merge instruments of the same program / inst before preprocessing them.
        # This allows to avoid potential duplicated notes in some multitrack settings
        # This can however mess up chord detections.
        if self.config.use_programs and self.one_token_stream:
            merge_same_program_tracks(midi.tracks)

        # Process time signature changes
        # We need to do it before computing the ticks_per_beat sections
        if self.config.use_time_signatures and len(midi.time_signatures) > 0:
            self._preprocess_time_signatures(midi.time_signatures)
        if len(midi.time_signatures) == 0:  # can sometimes happen
            midi.time_signatures.append(TimeSignature(0, *TIME_SIGNATURE))

        # Compute sections of the number of ticks per beat.
        # This is only used when using durations that need to be adjusted.
        if not self._note_on_off or (
            self.config.use_sustain_pedals and self.config.sustain_pedal_duration
        ):
            ticks_per_beat = get_midi_ticks_per_beat(midi)
        else:
            ticks_per_beat = None

        for t in range(len(midi.tracks) - 1, -1, -1):
            if len(midi.tracks[t].notes) == 0:
                del midi.tracks[t]
                continue
            # Preprocesses notes
            self._preprocess_notes(midi.tracks[t].notes, ticks_per_beat)

            if len(midi.tracks[t].notes) == 0:
                del midi.tracks[t]
                continue

            # Resample pitch bend values
            if self.config.use_pitch_bends and len(midi.tracks[t].pitch_bends) > 0:
                self._preprocess_pitch_bends(midi.tracks[t].pitch_bends)

            # Resample pedals durations
            if self.config.use_sustain_pedals and len(midi.tracks[t].pedals) > 0:
                self._preprocess_pedals(midi.tracks[t].pedals, ticks_per_beat)

        # Process tempo changes
        if self.config.use_tempos and len(midi.tempos) > 0:
            self._preprocess_tempos(midi.tempos)

        # We do not change key signature changes, markers and lyrics here as they are
        # not used by MidiTok (yet)

        return midi

    def _preprocess_notes(
        self, notes: NoteTickList, ticks_per_beat: np.ndarray = None
    ) -> None:
        r"""Resamples the note velocities, remove notes outside of pitch range.
        Note durations will be clipped to the maximum duration that can be handled by
        the tokenizer. This is done to prevent having incorrect offset values when
        computing rests. Notes with pitches outside of self.pitch_range will be
        deleted.

        :param notes: notes to preprocess.
        :param ticks_per_beat: array indicating the number of ticks per beat per time
            signature denominator section. The numbers of ticks per beat depend on the
            time signatures of the MIDI being parsed. The array has a shape ``(N,2)``,
            for ``N`` changes of ticks per beat, and the second dimension representing
            the end tick of each section and the number of ticks per beat respectively.
            This argument is not required if
            ``tokenizer.config.sustain_pedal_duration`` is disabled. (default: None)
        """
        # Gather times and velocity values in lists
        durations, velocities = [], []
        i = 0
        while i < len(notes):
            if (
                not self.config.pitch_range[0]
                <= notes[i].pitch
                < self.config.pitch_range[1]
            ):
                del notes[i]
                continue
            durations.append(notes[i].duration)
            velocities.append(notes[i].velocity)
            i += 1

        # Compute new velocities
        velocities = np_get_closest(self.velocities, np.array(velocities))
        for i, vel in enumerate(velocities):
            notes[i].velocity = vel

        # Compute new durations
        if not self._note_on_off:
            self._adjust_durations(notes, ticks_per_beat)

        # Symusic automatically sorts the notes by (time, pitch, duration) keys when
        # reading a MIDI file. We hence don't need to sort the notes.
        # However, when using `NoteOn`/`NoteOff`, we can encounter note order
        # alterations with the velocity values as they are not sorted on velocities and
        # that the tokens are decoded following a FIFO logic.
        # To alleviate this, a user can sort them before calling the tokenizer.
        # We do not do it here as it is not considered a disturbing issue, and that it
        # would add a significant overhead preprocessing time. This is however done in
        # the tokenization tests of MidiTok for concerned tokenizers in order to keep
        # 100% of the data integrity, so that the tests pass.

        if self.config.remove_duplicated_notes:
            remove_duplicated_notes(notes)

    def _preprocess_tempos(self, tempos: TempoTickList) -> None:
        r"""Resamples the tempo values of tempo change events.
        For tempo changes occurring at the same tick/time, we only keep the last one.
        Consecutive identical tempo changes will be removed if
        ``self.config.delete_equal_successive_tempo_changes`` is True.

        :param tempos: tempo changes to resample.
        """
        # If we delete the successive equal tempo changes, we need to sort them by time
        # Fortunately, sorting is already performed by symusic when loading the MIDI.

        # Gather times and velocity values in lists
        times, values = [], []
        for tempo in tempos:
            times.append(tempo.time)
            values.append(tempo.tempo)

        # Find the closest tempos
        times = np.array(times, dtype=np.intc)
        values = np_get_closest(self.tempos, np.array(values))

        # Find groups of tempos at the same onset ticks, equal consecutive ones
        if len(tempos) > 1:
            # Keep only last tempo change for groups with same tick
            idx_groups = np.split(
                np.arange(len(times)), np.where(np.diff(times) != 0)[0] + 1
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    for idx_to_del in reversed(idx_group[:-1]):
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
                            values = np.delete(values, idx_to_del)
                            del tempos[idx_to_del]

        # Apply new values
        for val, tempo in zip(values, tempos):
            tempo.tempo = val

    def _preprocess_time_signatures(self, time_sigs: TimeSignatureTickList) -> None:
        r"""Resamples the time signature changes.

        Time signature will be delayed to the next bar. See MIDI 1.0 Detailed
        specifications, pages 54 - 56, for more information on delayed time signature
        messages.
        If the ``delete_equal_successive_time_sig_changes`` parameter is set ``True``
        in the tokenizer's configuration, the time signatures must be sorted by time
        before calling this method. This is done by symusic when loading a MIDI. If
        this method is called for a MIDI created from another way, make sure they
        are sorted: ``midi.time_signatures.sort()``.

        :param time_sigs: time signature changes to quantize.
        """
        # Filter the first time signature
        while (
            len(time_sigs) > 0
            and (time_sigs[0].numerator, time_sigs[0].denominator)
            not in self.time_signatures
        ):
            if self._verbose:
                warnings.warn(
                    f"The MIDI contains a time signature ({time_sigs[0]}) outside of "
                    f"those supported by the tokenizer ({self.time_signatures}). You "
                    f"should either discard this MIDI or support this time signature, "
                    f"or alternatively deleting it however if you are using a "
                    f"beat-based tokenizer (REMI) the bars will be incorrectly "
                    f"detected.",
                    stacklevel=2,
                )
            del time_sigs[0]
            continue
        if len(time_sigs) == 0:
            return  # the default one will be added in `_preprocess_midi()`

        ticks_per_bar = compute_ticks_per_bar(time_sigs[0], self.time_division)
        previous_tick = 0  # first time signature change is always at tick 0
        prev_ts = (time_sigs[0].numerator, time_sigs[0].denominator)
        i = 1
        while i < len(time_sigs):
            time_sig = time_sigs[i]
            del_time_sig = False
            if (time_sig.numerator, time_sig.denominator) not in self.time_signatures:
                # Alternatively, we could offer a solution to "mock" unrecognized time
                # signatures. If one (not both) of the numerator or denominator value
                # is in the vocabulary, we could mock the other value with 4 (default).
                if self._verbose:
                    warnings.warn(
                        f"The MIDI contains a time signature ({time_sigs[i]}) outside "
                        f"of those supported by the tokenizer ({self.time_signatures})"
                        f". You should either discard this MIDI or support this time "
                        f"signature, or alternatively deleting it however if you are "
                        f"using a beat-based tokenizer (REMI) the bars will be "
                        f"incorrectly detected.",
                        stacklevel=2,
                    )
                del_time_sig = True
            if del_time_sig or (
                self.config.delete_equal_successive_time_sig_changes
                and (time_sig.numerator, time_sig.denominator) == prev_ts
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
            ticks_per_bar = compute_ticks_per_bar(time_sig, self.time_division)

            # If the current time signature is now at the same time as the previous
            # one, we delete the previous
            if time_sig.time == previous_tick:
                del time_sigs[i - 1]
                continue

            previous_tick = time_sig.time
            prev_ts = time_sig
            i += 1

    def _preprocess_pitch_bends(self, pitch_bends: PitchBendTickList) -> None:
        r"""Resamples the pitch bend events from a track.
        Overlapping pitch bends will be deduplicated by keeping the one
        having the highest absolute value at a given tick.

        :param pitch_bends: pitch bend events.
        """
        # Gather times and velocity values in lists
        times, values = [], []
        for pitch_bend in pitch_bends:
            times.append(pitch_bend.time)
            values.append(pitch_bend.value)

        # Resample time, remove 0 durations
        times = np.array(times, dtype=np.intc)
        values = np_get_closest(self.pitch_bends, np.array(values))

        # Find groups of pitch bends at the same onset ticks, and keep the > abs values
        if len(pitch_bends) > 1:
            idx_groups = np.split(
                np.arange(len(times)), np.where(np.diff(times) != 0)[0] + 1
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    values_group = values[idx_group]
                    max_abs_idx = np.argmax(np.abs(values_group))
                    values[idx_group[0]] = values_group[max_abs_idx]
                    for idx_to_del in reversed(idx_group[1:]):
                        values = np.delete(values, idx_to_del)
                        del pitch_bends[idx_to_del]

        # Apply new values
        for i, value in enumerate(values):
            pitch_bends[i].value = value

    def _preprocess_pedals(
        self, pedals: PedalTickList, ticks_per_beat: np.ndarray = None
    ) -> None:
        r"""Resamples the pedals durations.

        :param pedals: pedals to preprocess
        :param ticks_per_beat: array indicating the number of ticks per beat per
            portions. The numbers of ticks per beat depend on the time signatures of
            the MIDI being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively. This argument
            is not required if ``tokenizer.config.sustain_pedal_duration`` is disabled.
            (default: None)
        """
        # Get times and durations
        times_durations_ends = [[pd.time, pd.duration, pd.end] for pd in pedals]
        times_durations_ends = np.array(times_durations_ends, dtype=np.intc)

        # Reformat durations according to the tokenizer's vocabulary
        if self.config.sustain_pedal_duration:
            self._adjust_durations(pedals, ticks_per_beat)

        # Format durations (if needed) and merge successive pedals
        while np.any(times_durations_ends[:-1, 2] >= times_durations_ends[1:, 0]):
            # Merge PedalOn periods depending on their durations
            i = 1
            while i < len(pedals):
                if pedals[i - 1].end >= pedals[i].time:
                    pedals[i - 1].duration = max(
                        pedals[i - 1].duration, pedals[i - 1].time - pedals[i].end
                    )
                    del pedals[i]
                    times_durations_ends[i - 1, 1] = pedals[i - 1].duration
                    times_durations_ends[i - 1, 2] = pedals[i - 1].end
                    times_durations_ends = np.delete(times_durations_ends, i, axis=0)
                else:
                    i += 1

            # We need to readjust durations again as they may have changed after merge
            if self.config.sustain_pedal_duration:
                self._adjust_durations(pedals, ticks_per_beat)

    def _adjust_durations(
        self, notes_pedals: NoteTickList | PedalTickList, ticks_per_beat: np.ndarray
    ) -> None:
        """Adjust the durations of notes or pedals, to the closest ones of those in the
        tokenizer's vocabulary. The new durations are calculated depending on the time
        signature, i.e. number of ticks in a beat.

        :param notes_pedals: list of notes or pedals to process.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            portions. The numbers of ticks per beat depend on the time signatures of
            the MIDI being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
        """

        # Batch by tpb section
        dur_idx_first = 0
        for tpb_idx, (last_tick_tpb, tpb) in enumerate(ticks_per_beat):
            # Get idx of the concerned notes.
            # There shouldn't be equal successive tpb values in ticks_per_beat.
            # If last tpb --> set last note to max_tick to avoid iterating notes
            if tpb_idx + 1 == len(ticks_per_beat):
                dur_idx_last = len(notes_pedals)
                durations_section = [
                    obj_.duration for obj_ in notes_pedals[:dur_idx_last]
                ]
            else:
                dur_idx_last = 0  # excluded, so -1 in practice
                durations_section = []
                for obj_idx in range(dur_idx_first, len(notes_pedals)):
                    if notes_pedals[obj_idx].time >= last_tick_tpb:
                        dur_idx_last = obj_idx
                        break
                    durations_section.append(notes_pedals[obj_idx].duration)
            durations_section = np_get_closest(
                self._tpb_to_time_array[tpb], np.array(durations_section)
            )
            for dur_idx, obj_idx in enumerate(range(dur_idx_first, dur_idx_last)):
                notes_pedals[obj_idx].duration = durations_section[dur_idx]
            dur_idx_first = dur_idx_last

    def _midi_to_tokens(self, midi: Score) -> TokSequence | list[TokSequence]:
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

        # Compute ticks_per_beat sections depending on the time signatures
        # This has to be computed several times, in preprocess after resampling & here.
        ticks_per_beat = get_midi_ticks_per_beat(midi)

        # Adds track tokens
        for ti, track in enumerate(midi.tracks):
            track_events = self._create_track_events(track, ticks_per_beat)
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
            # Add ProgramChange (named Program) tokens if requested.
            if self.config.program_changes:
                self._insert_program_change_events(all_events)

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
        if event.type_ in ["Tempo", "TimeSig"]:
            return 0
        # Then NoteOff
        elif event.type_ == "NoteOff" or (
            event.type_ == "Program" and event.desc == "ProgramNoteOff"
        ):
            return 1
        # Then track effects
        elif event.type_ in ["Pedal", "PedalOff"] or (
            event.type_ == "Duration" and event.desc == "PedalDuration"
        ):
            return 2
        elif event.type_ == "PitchBend" or (
            event.type_ == "Program" and event.desc == "ProgramPitchBend"
        ):
            return 3
        elif event.type_ == "ControlChange":
            return 4
        # Track notes then
        else:
            return 10

    def _create_track_events(
        self, track: Track, ticks_per_beat: np.ndarray
    ) -> list[Event]:
        r"""Extract the tokens / events of individual tracks: *Pitch*, *Velocity*,
        *Duration*, *NoteOn*, *NoteOff* and optionally *Chord*, from a track
        (``symusic.Track``).
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in** ``preprocess_midi``.

        :param track: MIDI track to convert.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            portions. The numbers of ticks per beat depend on the time signatures of
            the MIDI being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
        :return: sequence of corresponding ``Event``s.
        """
        program = track.program if not track.is_drum else -1
        events = []
        note_token_name = "NoteOn" if self._note_on_off else "Pitch"
        # max_time_interval is adjusted depending on the time signature denom / tpb
        tpb_idx = 0
        max_time_interval = 0
        if self.config.use_pitch_intervals:
            max_time_interval = (
                ticks_per_beat[tpb_idx, 1] * self.config.pitch_intervals_max_time_dist
            )
        previous_note_onset = -max_time_interval - 1
        previous_pitch_onset = -128  # lowest at a given time
        previous_pitch_chord = -128  # for chord intervals

        # Add chords
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                ticks_per_beat,
                chord_maps=self.config.chord_maps,
                program=program,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_num_notes_range=self.config.chord_unknown,
            )
            for chord in chords:
                if self.config.use_programs and not self.config.program_changes:
                    events.append(
                        Event("Program", program, chord.time, program, "ProgramChord")
                    )
                events.append(chord)

        # Add sustain pedal
        if self.config.use_sustain_pedals:
            tpb_idx = 0
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
                    if pedal.time >= ticks_per_beat[tpb_idx, 0]:
                        tpb_idx += 1
                    dur = self._tpb_ticks_to_tokens[ticks_per_beat[tpb_idx, 1]][
                        pedal.duration
                    ]
                    events.append(
                        Event(
                            "Duration",
                            dur,
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
                        type_="Program",
                        value=program,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )

            # Pitch interval
            add_absolute_pitch_token = True
            if self.config.use_pitch_intervals and not track.is_drum:
                # Adjust max_time_interval if needed
                if note.time >= ticks_per_beat[tpb_idx, 0]:
                    tpb_idx += 1
                    max_time_interval = (
                        ticks_per_beat[tpb_idx, 1]
                        * self.config.pitch_intervals_max_time_dist
                    )
                if note.start != previous_note_onset:
                    if (
                        note.start - previous_note_onset <= max_time_interval
                        and abs(note.pitch - previous_pitch_onset)
                        <= self.config.max_pitch_interval
                    ):
                        events.append(
                            Event(
                                type_="PitchIntervalTime",
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
                                type_="PitchIntervalChord",
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

            # Pitch / NoteOn
            if add_absolute_pitch_token:
                events.append(
                    Event(
                        type_=note_token_name,
                        value=note.pitch,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )

            # Velocity
            events.append(
                Event(
                    type_="Velocity",
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
                            type_="Program",
                            value=program,
                            time=note.end,
                            program=program,
                            desc="ProgramNoteOff",
                        )
                    )
                events.append(
                    Event(
                        type_="NoteOff",
                        value=note.pitch,
                        time=note.end,
                        program=program,
                        desc=note.end,
                    )
                )
            else:
                if note.time >= ticks_per_beat[tpb_idx, 0]:
                    tpb_idx += 1
                dur = self._tpb_ticks_to_tokens[ticks_per_beat[tpb_idx, 1]][
                    note.duration
                ]
                events.append(
                    Event(
                        type_="Duration",
                        value=dur,
                        time=note.start,
                        program=program,
                        desc=f"{note.duration} ticks",
                    )
                )

        return events

    @staticmethod
    def _insert_program_change_events(events: list[Event]) -> None:
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
                and event.type_ not in ["Pedal", "PedalOff"]
                and not (event.type_ == "Duration" and previous_type == "Pedal")
            ):
                previous_program = event.program
                program_change_events.append(
                    (ei, Event("Program", event.program, event.time))
                )
            previous_type = event.type_

        for idx, event in reversed(program_change_events):
            events.insert(idx, event)

    def _create_midi_events(self, midi: Score) -> list[Event]:
        r"""Create the *global* MIDI additional tokens: `Tempo` and `TimeSignature`.

        :param midi: midi to extract the events from.
        :return: list of Events.
        """
        events = []

        # First adds time signature tokens if specified
        if self.config.use_time_signatures:
            events += [
                Event(
                    type_="TimeSig",
                    value=f"{time_sig.numerator}/" f"{time_sig.denominator}",
                    time=time_sig.time,
                )
                for time_sig in midi.time_signatures
            ]

        # Adds tempo events if specified
        if self.config.use_tempos:
            events += [
                Event(
                    type_="Tempo",
                    value=round(tempo.tempo, 2),  # req to handle c++ values
                    time=tempo.time,
                    desc=tempo.tempo,
                )
                for tempo in midi.tempos
            ]

        return events

    @abstractmethod
    def _add_time_events(self, events: list[Event]) -> list[Event]:
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
    ) -> TokSequence | list[TokSequence]:
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
        midi = self.preprocess_midi(midi)

        # Tokenize it
        tokens = self._midi_to_tokens(midi)
        if apply_bpe_if_possible and self.has_bpe:
            self.apply_bpe(tokens)

        return tokens

    def complete_sequence(self, seq: TokSequence) -> None:
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
        self, tokens: Sequence[str | list[str]]
    ) -> list[int | list[int]]:
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
        self, ids: list[int | list[int]], as_str: bool = True
    ) -> list[str | Event | list[str | Event]]:
        r"""Converts a sequence of ids (int) to their associated tokens (str or Event).
        **This method will not work with ids encoded with BPE. You will need to decode
        them first (:py:meth:`miditok.MIDITokenizer.decode_bpe`)**.

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
        events: list[Event | list[Event]],
    ) -> list[str | list[str]]:
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
        self, ids: list[int | list[int]], as_one_str: bool = False
    ) -> str | list[str]:
        r"""Converts a list of ids into their associated bytes.
        It can be returned either as a list of bytes or as a unique string of bytes.
        **This method will not work with ids encoded with BPE. You will need to decode
        them first (:py:meth:`miditok.MIDITokenizer.decode_bpe`)**.

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
        self, bytes_: str | list[str], as_str: bool = True
    ) -> list[str | Event | list[str | Event]]:
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

    def _convert_sequence_to_tokseq(
        self,
        input_seq: list[int | str | list[int | str]] | np.ndarray,
        complete_seq: bool = False,
        decode_bpe: bool = False,
    ) -> TokSequence | list[TokSequence]:
        r"""Converts a sequence into a :class:`miditok.TokSequence` or list of
        :class:`miditok.TokSequence` objects with the appropriate format of the
        tokenizer being used.

        :param input_seq: sequence to convert. It can be a list of ids (integers),
            tokens (string) or events (Event). It can also be a Pytorch or TensorFlow
            tensor, or Numpy array representing ids.
        :param complete_seq: will complete the output sequence(s). (default: ``False``)
        :param decode_bpe: if the input sequence contains ids, and that they contain
            BPE tokens, these tokens will be decoded. (default: ``False``)
        :return: the input sequence as a (list of) :class:`miditok.TokSequence`.
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

        # Deduce number of subscripts / dims
        num_io_dims = len(self.io_format)
        num_seq_dims = 1
        if len(arg[1]) > 0 and isinstance(arg[1][0], list):
            num_seq_dims += 1
            if len(arg[1][0]) > 0 and isinstance(arg[1][0][0], list):
                num_seq_dims += 1
            elif len(arg[1][0]) == 0 and num_seq_dims == num_io_dims - 1:
                # Special case where the sequence contains no tokens, we increment
                num_seq_dims += 1

        # Check the number of dimensions is good
        # In case of no one_token_stream and one dimension short --> unsqueeze
        if not self.one_token_stream and num_seq_dims == num_io_dims - 1:
            warnings.warn(
                f"The input sequence has one dimension less than expected ("
                f"{num_seq_dims} instead of {num_io_dims}). It is being unsqueezed to "
                f"conform with the tokenizer's i/o format ({self.io_format})",
                stacklevel=2,
            )
            arg = (arg[0], [arg[1]])

        elif num_seq_dims != num_io_dims:
            raise ValueError(
                f"The input sequence does not have the expected dimension "
                f"({num_seq_dims} instead of {num_io_dims})."
            )

        # Convert to TokSequence
        if not self.one_token_stream and num_io_dims == num_seq_dims:
            seq = []
            for obj in arg[1]:
                kwarg = {arg[0]: obj}
                seq.append(TokSequence(**kwarg))
                if not self.is_multi_voc and seq[-1].ids is not None:
                    seq[-1].ids_bpe_encoded = self._are_ids_bpe_encoded(seq[-1].ids)
        else:  # 1 subscript, one_token_stream and no multi-voc
            kwarg = {arg[0]: arg[1]}
            seq = TokSequence(**kwarg)
            if not self.is_multi_voc:
                seq.ids_bpe_encoded = self._are_ids_bpe_encoded(seq.ids)

        # decode BPE and complete the output sequence(s) if requested
        if self.has_bpe and decode_bpe:
            self.decode_bpe(seq)
        if complete_seq:
            if isinstance(seq, TokSequence):
                self.complete_sequence(seq)
            else:
                for seq_ in seq:
                    self.complete_sequence(seq_)

        return seq

    def _are_ids_bpe_encoded(self, ids: list[int] | np.ndarray) -> bool:
        r"""A small check telling if a sequence of ids are encoded with BPE.
        This is performed by checking if any id has a value superior or equal to the
        length of the base vocabulary.

        :param ids: ids to check.
        :return: boolean, True if ids are encoded with BPE, False otherwise.
        """
        return np.any(np.array(ids) >= len(self))

    def tokens_to_midi(
        self,
        tokens: TokSequence | list[TokSequence] | list[int | list[int]] | np.ndarray,
        programs: list[tuple[int, bool]] | None = None,
        output_path: str | None = None,
    ) -> Score:
        r"""Detokenize one or multiple sequences of tokens into a MIDI file.
        You can give the tokens sequences either as :class:`miditok.TokSequence`
        objects, lists of integers, numpy arrays or PyTorch / Tensorflow tensors.
        The MIDI's time division will be the same as the tokenizer's:
        ``tokenizer.time_division``.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence`, a Tensor (PyTorch and Tensorflow are
            supported), a numpy array or a Python list of ints. The first dimension
            represents tracks, unless the tokenizer handle tracks altogether as a
            single token sequence (``tokenizer.one_token_stream == True``).
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: None)
        :param output_path: path to save the file. (default: ``None``)
        :return: the midi object (``symusic.Score``).
        """

        if not isinstance(tokens, (TokSequence, list)) or (
            isinstance(tokens, list)
            and any(not isinstance(seq, TokSequence) for seq in tokens)
        ):
            tokens = self._convert_sequence_to_tokseq(
                tokens, complete_seq=True, decode_bpe=True
            )

        midi = self._tokens_to_midi(tokens, programs)

        # Set default tempo and time signatures at tick 0 if not present
        if len(midi.tempos) == 0 or midi.tempos[0].time != 0:
            midi.tempos.insert(0, Tempo(0, self.default_tempo))
        if len(midi.time_signatures) == 0 or midi.time_signatures[0].time != 0:
            midi.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump_midi(output_path)
        return midi

    @abstractmethod
    def _tokens_to_midi(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
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
        :return: the midi object (symusic.Score).
        """
        raise NotImplementedError

    @abstractmethod
    def _create_base_vocabulary(self) -> list[str | list[str]]:
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

    def __create_vocabulary(self) -> None:
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

    def _add_additional_tokens_to_vocab_list(self, vocab: list[str]) -> None:
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

    def _update_token_types_indexes(self) -> None:
        r"""Updates the _token_types_indexes attribute according to _event_to_token."""

        def create_for_dict(voc: dict[str, int]) -> dict[str, list[int]]:
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
        self, token_type: str, vocab_id: int | None = None
    ) -> list[int]:
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
        token: str | Event,
        vocab_idx: int | None = None,
        byte_: str | None = None,
        add_to_bpe_model: bool = False,
    ) -> None:
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

    def _create_chords_tokens(self) -> list[str]:
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

    def token_id_type(self, id_: int, vocab_id: int | None = None) -> str:
        r"""Returns the type of the given token id.

        :param id_: token id to get the type.
        :param vocab_id: index of the vocabulary associated to the token, if
            applicable. (default: None)
        :return: the type of the token, as a string
        """
        token = self.__get_from_voc(id_, vocab_id)
        return token.split("_")[0]

    @abstractmethod
    def _create_token_types_graph(self) -> dict[str, list[str]]:
        r"""Creates a dictionary describing the possible token type successions.
        This method is unimplemented and need to be overridden by inheriting classes.
        See other classes (:class:`miditok.REMI._create_token_types_graph`, ...)
        for examples of how to implement it.
        """
        raise NotImplementedError

    def _add_special_tokens_to_types_graph(self) -> None:
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

    def __create_durations_tuples(self) -> list[tuple[int, int, int]]:
        r"""Creates the possible durations in beat / position units, as tuple of the
        form: (beat, pos, res) where beat is the number of beats, pos the number of
        "samples" and res the beat resolution considered (samples per beat).
        Example: (2, 5, 8) means the duration is 2 beat long + position 5 / 8 of the
        ongoing beat. This would give in ticks:
        ``duration = (beat * res + pos) * ticks_per_beat // res``
        Note that ``ticks_per_beat`` is different from the time division, as the number
        of ticks per beat depends on the current time signature denominator.
        If ticks_per_beat is 384:
        ``duration = (2 * 8 + 5) * 384 // 8 = 1008`` ticks.

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

    @property
    def __tpb_set(self) -> set[int]:
        """Returns the set of the possible ticks per beat resolution covered by the
        time signatures of the tokenizer.

        :return: set of ticks per beat.
        """
        return {
            compute_ticks_per_beat(denom, self.time_division)
            for denom in {ts[1] for ts in self.time_signatures}
        }

    def __create_tpb_to_ticks_array(self, rest: bool = False) -> dict[int, np.ndarray]:
        r"""Creates arrays of the time values in ticks of the time tokens of the
        tokenizer, depending on the ticks per beat (time signature).
        The arrays are used during tokenization to efficiently find the closest values.

        :param rest: will use rest values if given ``True``, otherwise durations.
            (default: ``False``)
        :return: dictionary of the durations in tick depending on the ticks per beat
            resolution.
        """
        values = self.rests if rest else self.durations
        return {
            tpb: np.array(
                [self._time_token_to_ticks(time_tuple, tpb) for time_tuple in values],
                dtype=np.intc,
            )
            for tpb in self.__tpb_set
        }

    def __create_tpb_tokens_to_ticks(
        self, rest: bool = False
    ) -> dict[int, dict[int, str]]:
        r"""Creates the correspondences between duration in tick and token value (str
        in beats and samples) for the "ticks per beat" resolutions covered by the
        tokenizer.

        The returned dictionary is used when decoding *Duration*/*TimeShift*/*Rest*
        tokens while taking the time signature into account.

        :param rest: will use rest values if given ``True``, otherwise durations.
            (default: ``False``)
        :return: ticks per beat + token value to duration in tick.
        """
        values = self.rests if rest else self.durations
        return {
            tpb: {
                ".".join(map(str, duration_tuple)): self._time_token_to_ticks(
                    duration_tuple, tpb
                )
                for duration_tuple in values
            }
            for tpb in self.__tpb_set
        }

    def __create_tpb_ticks_to_tokens(self) -> dict[int, dict[int, str]]:
        r"""Creates the correspondences between duration in tick and token value (str
        in beats and samples) for the "ticks per beat" resolutions covered by the
        tokenizer.

        The returned dictionary is used during tokenization to get the values of
        *Duration*/*TimeShift*/*Rest* tokens while taking the time signature into
        account.

        :return: ticks per beat + duration in ticks to token value.
        """
        return {
            tpb: {v: k for k, v in tokens_to_ticks.items()}
            for tpb, tokens_to_ticks in self._tpb_tokens_to_ticks.items()
        }

    @staticmethod
    def _time_token_to_ticks(
        token_duration: str | tuple[int, int, int], ticks_per_beat: int
    ) -> int:
        r"""Converts a time token value of the form x.x.x, for
        beat.position.resolution, in ticks. This method is used to decode time tokens
        such as *Duration*, *TimeShift* or *Rest*.

        :param token_duration: Duration / TimeShift token value.
        :param ticks_per_beat: number of ticks in a beat. This depends on the current
            time signature, and is equal to the MIDI's time division if the denominator
            is 4 (quarter).
        :return: the duration / time-shift in ticks.
        """
        if isinstance(token_duration, str):
            token_duration = tuple(map(int, token_duration.split(".")))
        beat, pos, res = token_duration
        # We don't need to round anything here, as `ticks_per_beat` is always divisible
        # by `res`.
        return (beat * res + pos) * ticks_per_beat // res

    def _time_ticks_to_tokens(
        self,
        duration: int,
        ticks_per_beat: int,
        rest: bool = False,
    ) -> tuple[list[tuple[int, int, int]], list[int]]:
        r"""Converts a duration in ticks into a sequence of *TimeShift*/*Rest* values.
        This method is not used for *Duration* tokens, as their values are rounded to
        the closest values in

        :param duration: duration in tick to convert.
        :param ticks_per_beat: number of ticks in a beat. This depends on the current
            time signature, and is equal to the MIDI's time division if the denominator
            is 4 (quarter).
        :param rest: the duration is a rest, hence the created tokens will be based on
            the ``self.rests`` values.
        :return: list of associated token values, and the list of the elapsed offset in
            tick for each of these values.
        """
        if rest:
            time_bins = self._tpb_to_rest_array[ticks_per_beat]
            time_tokens = self.rests
        else:
            time_bins = self._tpb_to_time_array[ticks_per_beat]
            time_tokens = self.durations
        min_time = time_bins[0]

        offset_times, values = [], []
        while duration >= min_time:
            index = (time_bins - duration <= 0).nonzero()[0][-1]
            values.append(time_tokens[index])
            val_ticks = time_bins[index]
            duration -= val_ticks
            offset_times.append(int(val_ticks))

        return values, offset_times

    def __create_rests(self) -> list[tuple[int, int, int]]:
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

        The self.config.num_tempos tempos are distributed in the self.config.tempo_range
        using either log or linear scaled values based on the value of
        ``self.config.log_tempos``.

        :return: the tempos.
        """
        tempo_fn = np.geomspace if self.config.log_tempos else np.linspace
        return tempo_fn(*self.config.tempo_range, self.config.num_tempos).round(2)

    def __create_time_signatures(self) -> list[tuple]:
        r"""Creates the possible time signatures, as tuples of the form:
        ``(num_beats, beat_res)`` where ``num_beats`` is the number of beats per bar.
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

            time_signatures.extend([(num_beats, beat_res) for num_beats in beats])

        return time_signatures

    def __create_pitch_bends(self) -> np.ndarray:
        r"""Creates the pitch bend values, as numpy array, using
        ``self.config.pitch_bend_range``.

        :return: the pitch bend values.
        """
        return np.linspace(*self.config.pitch_bend_range, dtype=np.int32)

    @staticmethod
    def _parse_token_time_signature(token_time_sig: str) -> tuple[int, int]:
        r"""Converts a time signature token value of the form x/x into a tuple of
        integers, time signature's numerator (bar length in beats) and denominator
        (beat resolution).

        :param token_time_sig: TimeSig token value.
        :return: the numerator and denominator of a time signature.
        """
        numerator, denominator = map(int, token_time_sig.split("/"))
        return numerator, denominator

    def has_midi_time_signatures_not_in_vocab(self, midi: Score) -> bool:
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
        iterator: Iterable | None = None,
        files_paths: list[Path | str] | None = None,
        start_from_empty_voc: bool = False,
        **kwargs,
    ) -> None:
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
        :param kwargs: any additional argument to pass to the trainer. See the
            `tokenizers docs <https://huggingface.co/docs/tokenizers/api/trainers>`_
            for more details.
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
            num_bytes = (
                len(self.config.special_tokens)
                if start_from_empty_voc
                else len(self._vocab_base)
            )
            voc_start = {chr(i + CHR_ID_START): i for i in range(num_bytes)}
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
            new_vocab = dict(
                sorted(self._bpe_model.get_vocab().items(), key=lambda item: item[1])
            )
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

    def apply_bpe(self, seq: TokSequence | list[TokSequence]) -> None:
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

    def decode_bpe(self, seq: TokSequence | list[TokSequence]) -> None:
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
        midi_paths: str | Path | Sequence[str | Path],
        out_dir: str | Path,
        overwrite_mode: bool = True,
        validation_fn: Callable[[Score], bool] | None = None,
        save_programs: bool | None = None,
        verbose: bool = True,
    ) -> None:
        r"""Converts a dataset / list of MIDI files, into their token version and save
        them as json files. The resulting json files will have an ``ids`` entry
        containing the token ids. The format of the ids will correspond to the format
        of the tokenizer (``tokenizer.io_format``). Note that the file tree of the
        source files, up to the deepest common root directory if `midi_paths` is given
        as a list of paths, will be reproducing in ``out_dir``. The config of the
        tokenizer will be saved as a file named ``tokenizer_config_file_name``
        (default: ``tokenizer.json``) in the ``out_dir`` directory.

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
            (default: ``True``)
        :param validation_fn: a function checking if the MIDI is valid on your
            requirements (e.g. time signature, minimum/maximum length, instruments...).
            (default: ``None``)
        :param save_programs: will save the programs of the tracks of the MIDI as an
            entry in the Json file. That this option is probably unnecessary when using
            a multitrack tokenizer (`config.use_programs`), as the program information
            is present within the tokens, and that the tracks having the same programs
            are likely to have been merged. (default: ``False`` if
            ``config.use_programs``, else ``True``)
        :param verbose: will throw warnings of errors when loading MIDI files, or if
            some MIDI content is incorrect or need your attention. (default: ``True``)
        """
        self._verbose = verbose
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
                if len({parts[depth] for parts in all_parts}) > 1:
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
                if self._verbose:
                    warnings.warn(f"File not found: {midi_path}", stacklevel=2)
                continue
            except MIDI_LOADING_EXCEPTION:
                continue

            # Passing the MIDI to validation tests if given
            if validation_fn is not None and not validation_fn(midi):
                continue

            # Tokenizing the MIDI
            tokens = self.midi_to_tokens(midi)

            # Set output file path
            out_path = out_dir / midi_path.parent.relative_to(root_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            out_path /= f"{midi_path.stem}.json"

            # If non-overwrite, set the new file name
            if not overwrite_mode and out_path.is_file():
                i = 1
                while out_path.is_file():
                    out_path = out_path.parent / f"{midi_path.stem}_{i}.json"
                    i += 1

            # Save the tokens as JSON
            self.save_tokens(
                tokens,
                out_path,
                get_midi_programs(midi) if save_programs else None,
            )

        # Set it back to False
        self._verbose = False

    def tokens_errors(
        self,
        tokens: TokSequence | list[TokSequence] | list[int | list[int]] | np.ndarray,
    ) -> float | list[float]:
        r"""Checks if a sequence of tokens is made of good token types successions and
        returns the error ratio (lower is better).

        :param tokens: sequence of tokens to check.
        :return: the error ratio (lower is better).
        """
        if not isinstance(tokens, (TokSequence, list)) or (
            isinstance(tokens, list)
            and any(not isinstance(seq, TokSequence) for seq in tokens)
        ):
            tokens = self._convert_sequence_to_tokseq(tokens)

        # If list of TokSequence -> recursive
        if isinstance(tokens, list):
            return [self.tokens_errors(tok_seq) for tok_seq in tokens]
        elif len(tokens) == 0:
            return 0

        num_tok_predicted = len(tokens)  # used to norm the score
        if self.has_bpe:
            self.decode_bpe(tokens)
        self.complete_sequence(tokens)

        # Compute number of errors and norm by number of tokens predicted
        return self._tokens_errors(tokens.tokens) / num_tok_predicted

    def _tokens_errors(self, tokens: list[str | list[str]]) -> int:
        r"""Checks if a sequence of tokens is made of good token types successions and
        returns the error ratio (lower is better). This method receives a list of
        tokens as a list of strings, and returns the absolute number of errors
        predicted. The number of errors should not be higher than the number of tokens.
        This method is intended to be subclasses by tokenizer classes. The
        implementation in ``MIDITokenizer`` class will check token types, duplicated
        notes and time errors. It works for ``REMI``, ``TSD`` and ``Structured``.

        :param tokens: sequence of tokens string to check.
        :return: the number of errors predicted (no more than one per token).
        """
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
                    if (
                        self.config.remove_duplicated_notes
                        and pitch_val in current_pitches[current_program]
                    ):
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

        return err_type + err_time + err_note

    def save_tokens(
        self,
        tokens: TokSequence | list[int] | np.ndarray,
        path: str | Path,
        programs: list[tuple[int, bool]] | None = None,
        **kwargs,
    ) -> None:
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
    def load_tokens(path: str | Path) -> dict[str, list[int]]:
        r"""Loads tokens saved as JSON files.

        :param path: path of the file to load.
        :return: the tokens, with the associated information saved with.
        """
        with Path(path).open() as file:
            return json.load(file)

    def _save_pretrained(self, *args, **kwargs) -> None:  # noqa: ANN002
        # called by `ModelHubMixin.from_pretrained`.
        self.save_params(*args, **kwargs)

    def save_params(
        self,
        out_path: str | Path,
        additional_attributes: dict | None = None,
        filename: str | None = DEFAULT_TOKENIZER_FILE_NAME,
    ) -> None:
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
            to a directory. (default: ``"tokenizer.json"``)
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
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: dict | None,
        resume_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **kwargs,
    ) -> MIDITokenizer:
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

    def _load_params(self, config_file_path: str | Path) -> None:
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
    def io_format(self) -> tuple[str]:
        format_ = []
        if not self.one_token_stream:
            format_.append("I")
        format_.append("T")
        if self.is_multi_voc:
            format_.append("C")

        return tuple(d for d in format_)

    def __call__(
        self,
        obj: Score | TokSequence | list[TokSequence, int, list[int]],
        *args,  # noqa: ANN002
        **kwargs,
    ) -> TokSequence | list[TokSequence] | Score:
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
                return self.tokens_to_midi(tokens["ids"], *args, **kwargs)

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
    def len(self) -> int | list[int]:  # noqa: A003
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling/have multiple vocabularies, it will return the **list** of their
        lengths. Use the :py:func:`miditok.MIDITokenizer.__len__` magic method
        (``len(tokenizer)``) to get the sum of the lengths.

        :return: length of the vocabulary.
        """
        return [len(v) for v in self.vocab] if self.is_multi_voc else len(self)

    def __repr__(self) -> str:
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
        self, item: int | str | tuple[int, int | str]
    ) -> str | int | list[int]:
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

    def __get_from_voc(self, item: int | str, vocab_id: int | None = None) -> int | str:
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

    def __eq__(self, other: MIDITokenizer) -> bool:
        r"""Checks if two tokenizers are identical. This is done by comparing their
        vocabularies, and configuration.

        :param other: tokenizer to compare.
        :return: True if the vocabulary(ies) are identical, False otherwise.
        """
        if not isinstance(other, MIDITokenizer):
            return False
        bpe_voc_eq = True
        if self._bpe_model is not None and other._bpe_model is not None:
            bpe_voc_eq = self._bpe_model.get_vocab() == other._bpe_model.get_vocab()
        return (
            self._vocab_base == other._vocab_base
            and bpe_voc_eq
            and self._vocab_base_byte_to_token == other._vocab_base_byte_to_token
            and self.config == other.config
        )
