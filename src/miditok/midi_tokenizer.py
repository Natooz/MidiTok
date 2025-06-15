"""Base tokenizer class, acting as a "framework" for all tokenizers."""

from __future__ import annotations

import json
import math
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    TimeSignature,
    Track,
)
from symusic.core import (
    NoteTickList,
    PedalTickList,
    PitchBendTickList,
    ScoreTick,
    TimeSignatureTickList,
)

try:
    from miditoolkit import MidiFile
except ImportError:
    MidiFile = None
from tokenizers import AddedToken
from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders as _decoders
from tokenizers import models as _tok_models
from tokenizers import pre_tokenizers as _pre_tokenizers
from tokenizers import trainers as _tok_trainers
from tqdm import tqdm

from .attribute_controls import (
    BarAttributeControl,
    BarNoteDensity,
    BarNoteDuration,
    BarOnsetPolyphony,
    BarPitchClass,
    TrackNoteDensity,
    TrackNoteDuration,
    TrackOnsetPolyphony,
    TrackRepetition,
)
from .classes import Event, TokenizerConfig, TokSequence
from .constants import (
    ABC_FILES_EXTENSIONS,
    BOS_TOKEN_NAME,
    CHR_ID_START,
    CURRENT_MIDITOK_VERSION,
    CURRENT_SYMUSIC_VERSION,
    CURRENT_TOKENIZERS_VERSION,
    DEFAULT_TOKENIZER_FILE_NAME,
    DEFAULT_TRAINING_MODEL_NAME,
    EOS_TOKEN_NAME,
    PITCH_CLASSES,
    SCORE_LOADING_EXCEPTION,
    SUPPORTED_MUSIC_FILE_EXTENSIONS,
    TEMPO,
    TIME_SIGNATURE,
    TOKEN_TYPE_BEFORE_PC,
    UNIGRAM_MAX_INPUT_CHARS_PER_WORD_BAR,
    UNIGRAM_MAX_INPUT_CHARS_PER_WORD_BEAT,
    UNIGRAM_SPECIAL_TOKEN_SUFFIX,
    UNKNOWN_CHORD_PREFIX,
    WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BAR,
    WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BEAT,
)
from .tokenizer_training_iterator import TokTrainingIterator
from .utils import (
    compute_ticks_per_bar,
    convert_ids_tensors_to_list,
    detect_chords,
    get_bars_ticks,
    get_beats_ticks,
    get_score_programs,
    get_score_ticks_per_beat,
    is_track_empty,
    merge_same_program_tracks,
    remove_duplicated_notes,
)
from .utils.utils import (
    add_bar_beats_ticks_to_tokseq,
    get_deepest_common_subdir,
    miditoolkit_to_symusic,
    np_get_closest,
    tempo_qpm_to_mspq,
)

if TYPE_CHECKING:
    from symusic.core import TempoTickList

    from .attribute_controls import AttributeControl


class MusicTokenizer(ABC, HFHubMixin):
    r"""
    Base music tokenizer class, acting as a common framework.

    This is the base class of all tokenizers, containing the common methods and
    attributes. It serves as a framework, and implement most of the tokenization
    global workflow. Child classes should only implement specific methods, for their
    specific behaviors, leaving most of the logic here.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.TokenizerConfig` object.
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful if
        the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str | Path | None = None,
    ) -> None:
        # vocab of prime tokens, can be viewed as unique char / bytes
        self._vocab_base = {}
        # the other way, to decode id (int) -> token (str)
        self.__vocab_base_inv = {}
        # id (int) -> byte (str), as this might not be chr(id) after tokenizer training
        self._vocab_base_id_to_byte = {}
        # byte (str) -> token (str), for basic tokens
        self._vocab_base_byte_to_token = {}
        # byte(s) -> token(s), for faster BPE/Unigram/WordPiece decoding
        self._vocab_learned_bytes_to_tokens = {}
        # Fast tokenizer model backed with 🤗tokenizers
        self._model = None
        # Used in _notes_to_events, especially MIDILike
        self._note_on_off = False

        # Initialize config
        # Loading params, or initializing them from args
        if params is not None:
            # Will overwrite self.config
            self._load_from_json(params)
        # If no TokenizerConfig is given, we falls back to the default parameters
        elif tokenizer_config is None:
            self.config = TokenizerConfig()
        else:
            self.config = tokenizer_config

        # Tweak the tokenizer's configuration and / or attributes before creating the
        # vocabulary. This method is intended to be overridden by inheriting tokenizer
        # classes.
        self._tweak_config_before_creating_voc()

        # Determines whether the tokenizer will produce a single sequence of tokens for
        # all the tracks one token sequence per track. This is attribute is distinct
        # from `config.one_token_stream_for_programs` as they might not be equal (e.g.
        # MMM). `tokenizer.one_token_stream` is meant to be used for i/o format
        # purposes whereas `config.one_token_stream_for_programs` is used for Score
        # preprocessing (merging tracks or not) and tokenization (all tracks at once or
        # treated separately).
        self.one_token_stream = (
            self.config.use_programs and self.config.one_token_stream_for_programs
        )

        # Time Signatures
        # Need to be set before creating duration values/tokens.
        self.time_signatures = [TIME_SIGNATURE]
        if self.config.use_time_signatures:
            self.time_signatures = self.__create_time_signatures()

        # Creates the numbers of ticks/beats depending on the note values (time
        # signature denominator) of the tokenizer. The values are calculated depending
        # on the maximum number of positions per beat
        # (`self.config.max_num_pos_per_beat`) and the time signatures it supports.
        # The highest note value will ba assigned a tick/beat value equal to
        # `self.config.max_num_pos_per_beat`, the other note values will have a
        # ticks/beat based on this base, i.e. `self.config.max_num_pos_per_beat`
        # multiplied by the factor between each note value and the maximum note value.
        self._tpb_per_ts = self.__create_tpb_per_ts()
        # Default time division to use when decoding tokens to a ``symusic.Score``.
        # This is left as a class attribute and not a property as the config is not
        # intended to be modified after its creation. Ultimately, this could be
        # ensured by converting TokenizerConfig to a frozen dataclass.
        # It shouldn't be used in place of the real ticks/beat value, which depends on
        # the current time signature denominator. The only exception is for tokenizers
        # which does not support time signature, i.e. which only consider 4/4.
        # During preprocessing, the Score will be resampled to a new time division equal
        # to the number of ticks per beat of the lowest time signature denominator it
        # contains. This is done in order to resample as much as possible while keeping
        # most of the accuracy. Some sections might need to be resampled again, when
        # the time signature denominator will be higher (i.e. higher number of absolute
        # ticks per beat).
        # Related: https://github.com/Yikai-Liao/symusic/issues/10
        self.time_division = self._tpb_per_ts[TIME_SIGNATURE[1]]

        # Durations
        # Usages:
        # Duration: tpb --> np.array (ticks) to get the closest;
        # Duration/TimeShift/Rest: ticks + tpb --> token (str);
        # Duration/TimeShift/Rest: token + tpb --> ticks (int);
        self.durations = self._create_durations_tuples()
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
        if self.config.use_velocities:
            self.velocities = np.linspace(
                0, 127, self.config.num_velocities + 1, dtype=np.intc
            )[1:]
        else:
            self.velocities = np.array([])
        self._first_beat_res = next(iter(self.config.beat_res.values()))
        for beat_range, res in self.config.beat_res.items():
            if 0 in beat_range:
                self._first_beat_res = res
                break

        # Tempos
        # _DEFAULT_TEMPO is the closest one to 120 that the tokenizer supports
        self.tempos = np.zeros(1)
        self._tempos_mspq = np.zeros(1)
        self.default_tempo = TEMPO
        if self.config.use_tempos:
            self.tempos = self.__create_tempos()
            # _tempos_mspq is only used in preprocess_tempos to adjust to the closest
            self._tempos_mspq: np.ndarray = tempo_qpm_to_mspq(self.tempos)
            self._tempos_mspq.sort()
            self.default_tempo = self.tempos[np.argmin(np.abs(self.tempos - TEMPO))]

        # Pitch bends
        self.pitch_bends = np.zeros(1)
        if self.config.use_pitch_bends:
            self.pitch_bends = self.__create_pitch_bends()

        # Vocabulary and token types graph
        # The vocabulary might have already been created if the tokenizer is being
        # loaded.
        if len(self.vocab) == 0:
            self.__create_vocabulary()

        # Attribute controls
        self.attribute_controls = []
        if self.config.one_token_stream_for_programs or self.is_multi_voc:
            self._disable_attribute_controls()
            warnings.warn(
                "Attribute controls are not compatible with "
                "'config.one_token_stream_for_programs' and multi-vocabulary "
                "tokenizers. Disabling them from the config.",
                stacklevel=2,
            )
        else:
            self._initialize_attribute_controls()

        # Token type graph
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

    def _initialize_attribute_controls(self) -> None:
        if self.config.ac_polyphony_track:
            self.add_attribute_control(
                TrackOnsetPolyphony(
                    self.config.ac_polyphony_min, self.config.ac_polyphony_max
                )
            )
        if self.config.ac_polyphony_bar:
            self.add_attribute_control(
                BarOnsetPolyphony(
                    self.config.ac_polyphony_min, self.config.ac_polyphony_max
                )
            )
        if self.config.ac_pitch_class_bar:
            self.add_attribute_control(BarPitchClass())
        if self.config.ac_note_density_track:
            self.add_attribute_control(
                TrackNoteDensity(
                    self.config.ac_note_density_track_min,
                    self.config.ac_note_density_track_max,
                )
            )
        if self.config.ac_note_density_bar:
            self.add_attribute_control(
                BarNoteDensity(self.config.ac_note_density_bar_max)
            )
        if self.config.ac_note_duration_bar:
            self.add_attribute_control(BarNoteDuration())
        if self.config.ac_note_duration_track:
            self.add_attribute_control(TrackNoteDuration())
        if self.config.ac_repetition_track:
            self.add_attribute_control(
                TrackRepetition(
                    self.config.ac_repetition_track_num_bins,
                    self.config.ac_repetition_track_num_consec_bars,
                    self.config.pitch_range,
                )
            )

    def add_attribute_control(self, attribute_control: AttributeControl) -> None:
        """
        Add a :class:`miditok.attribute_control.AttributeControl` to the tokenizer.

        The tokens of the attribute controls will also be added to the vocabulary.

        :param attribute_control: :class:`miditok.attribute_control.AttributeControl` to
            add to the tokenizer.
        """
        self.attribute_controls.append(attribute_control)
        for token in attribute_control.tokens:
            if token not in self.vocab:
                self.add_to_vocab(token)

    @property
    def pad_token_id(self) -> int:
        """
        Return the id of the padding token (``PAD_None``). It is usually 0.

        :return: id of the padding token (``PAD_None``).
        """
        return (
            self._vocab_base["PAD_None"]
            if not self.is_multi_voc
            else self._vocab_base[0]["PAD_None"]
        )

    @property
    def vocab(
        self,
    ) -> dict[str, int] | list[dict[str, int]]:  # token (str) to its id (int)
        """
        Get the base vocabulary, as a dictionary mapping tokens (str) to their ids.

        The different (hidden / protected) vocabulary attributes of the class are:

        * ``._vocab_base`` : Dict[str: int] token -> id - Registers all known base
            tokens;
        * ``.__vocab_base_inv`` : Dict[int: str] id -> token - Inverse of
            ``._base_vocab`` , to go the other way;
        * ``._vocab_base_id_to_byte`` : Dict[int: str] id -> byte - Link ids to their
            associated unique bytes;
        * ``._vocab_base_byte_to_token`` : Dict[str: str] - similar as above but for
            tokens;
        * ``._vocab_learned_bytes_to_tokens`` : Dict[str: List[str]] byte(s) -> token(s)
            used to decode BPE/Unigram/WordPiece token ids;
        * ``._model.get_vocab()`` : Dict[str: int] byte -> id - BPE/Unigram/WordPiece
            model vocabulary, based on unique bytes.

        Before training the tokenizer, bytes are obtained by running ``chr(id)`` .
        After training, if we did start from an empty vocabulary, some base tokens might
        be removed from ``._vocab_base`` , if they were never found in the training
        samples. The base vocabulary being changed, ``chr(id)`` would then bind to
        incorrect bytes (on which byte succession would not have been learned). We
        register the original id/token/byte association in ``._vocab_base_id_to_byte``
        and ``._vocab_base_byte_to_token`` .

        :return: the base vocabulary.
        """
        return self._vocab_base

    @property
    def vocab_model(self) -> None | dict[str, int]:  # byte (str) to its id (int)
        r"""
        Return the vocabulary learnt with BPE.

        In case the tokenizer has not been trained with BPE, it returns None.

        :return: the BPE model's vocabulary.
        """
        if not self.is_trained:
            return None
        return self._model.get_vocab()

    @property
    def _model_name(self) -> str:
        if self._model is None:
            return "None"
        return type(self._model.model).__name__

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary, by calling ``len(tokenizer)``.

        :return: size of the vocabulary.
        """
        return len(self)

    @property
    def special_tokens(self) -> list[str]:
        r"""
        Return the special tokens in the vocabulary.

        :return: special tokens of the tokenizer
        """
        return self.config.special_tokens

    @property
    def special_tokens_ids(self) -> list[int]:
        r"""
        Return the ids of the special tokens in the vocabulary.

        :return: ids of the special tokens of the tokenizer
        """
        return [self[token] for token in self.special_tokens]

    @property
    def is_trained(self) -> bool:
        """
        Indicate if the tokenizer is trained (``True``).

        :return: a boolean, equal to ``True`` if the tokenizer is trained, ``False``
            otherwise.
        """
        return self._model is not None

    def _min_rest(self, ticks_per_beat: int) -> int:
        """
        Return the minimum rest value in ticks, for a given ``ticks_per_beat``.

        :param ticks_per_beat: number of ticks in a beat. This depends on the current
            time signature, and is equal to the Score's time division if the denominator
            is 4 (quarter).
        :return: minimum rest in ticks.
        """
        if not self.config.use_rests:
            return 0
        return int(self._tpb_to_rest_array[ticks_per_beat][0])

    def preprocess_score(self, score: Score) -> Score:
        r"""
        Pre-process a ``symusic.Score`` object to resample its time and events values.

        This method is called before parsing a Score's contents for tokenization.
        Its notes attributes (times, pitches, velocities) will be downsampled and
        sorted, duplicated notes removed, as well as tempos. Empty tracks (with no
        note) will be removed from the ``symusic.Score`` object. Notes with pitches
        outside ``self.config.pitch_range`` will be deleted. Tracks with programs not
        supported by the tokenizer will be deleted.

        This method is **not inplace** and does not alter the provided ``score`` object.

        :param score: ``symusic.Score`` object to preprocess.
        :return: the preprocessed ``score``.
        """
        # Filter time signatures.
        # We need to do this first to determine the Score's new time division.
        # A copy of the time signatures is made here to make inplace operations without
        # modifying the provided Score object. This copy will be set to the copy of the
        # score after resampling it.
        time_signatures_copy = score.time_signatures.copy()
        if self.config.use_time_signatures:
            self._filter_unsupported_time_signatures(time_signatures_copy)
            # We mock the first with 0, even if there are already time signatures. This
            # is required as if the Score only had */2 time signatures, we must make
            # sure the resampling tpq is calculated according to a maximum denom of 4
            # if the beginning of the Score is mocked at 4/4.
            if len(time_signatures_copy) == 0 or time_signatures_copy[0].time != 0:
                time_signatures_copy.insert(0, TimeSignature(0, *TIME_SIGNATURE))
            # The new time division is chosen depending on its highest time signature
            # denominator, and is equivalent to the highest possible tick/beat ratio.
            max_ts_denom = max(ts.denominator for ts in time_signatures_copy)
            new_tpq = int(self.config.max_num_pos_per_beat * max_ts_denom / 4)
        else:
            time_signatures_copy = TimeSignatureTickList(
                [TimeSignature(0, *TIME_SIGNATURE)]
            )
            new_tpq = self.config.max_num_pos_per_beat

        # Resample time if needed (not inplace) and attribute preprocessed time sig.
        score = self._resample_score(score, new_tpq, time_signatures_copy)

        # Merge instruments of the same program / inst before preprocessing them.
        # This allows to avoid potential duplicated notes in some multitrack settings
        # This can however mess up chord detections.
        if self.config.use_programs and self.config.one_token_stream_for_programs:
            merge_same_program_tracks(score.tracks)

        # Process time signature changes
        # We need to do it before computing the ticks_per_beat sections
        if self.config.use_time_signatures and len(score.time_signatures) > 0:
            self._preprocess_time_signatures(
                score.time_signatures, score.ticks_per_quarter
            )

        # Compute resampling ratios to update times of events when several time sig,
        # and ticks per beat ratios.
        # Resampling factors are used to resample times of events when the Score has
        # several different time signature denominators.
        # ticks_per_beat ratios are used to adjust durations values according to the
        # tokenizer's vocabulary, i.e. *Duration* tokens.
        if not self._note_on_off or (
            self.config.use_sustain_pedals and self.config.sustain_pedal_duration
        ):
            if self.config.use_time_signatures and len(score.time_signatures) > 0:
                ticks_per_beat = get_score_ticks_per_beat(score)
            else:
                ticks_per_beat = np.array([[score.end(), score.ticks_per_quarter]])
        else:
            ticks_per_beat = None
        if (
            self.config.use_time_signatures
            and len({ts.denominator for ts in score.time_signatures}) > 1
        ):
            tpq_resampling_factors = self._get_score_resampling_factor(score)
        else:
            tpq_resampling_factors = None

        # Preprocess track events
        for t in range(len(score.tracks) - 1, -1, -1):
            # Delete track only there is nothing inside being used
            program = -1 if score.tracks[t].is_drum else score.tracks[t].program
            if is_track_empty(
                score.tracks[t],
                check_pedals=self.config.use_sustain_pedals,
                check_pitch_bend=self.config.use_pitch_bends,
            ) or (self.config.use_programs and program not in self.config.programs):
                del score.tracks[t]
                continue

            # Preprocesses notes
            if len(score.tracks[t].notes) > 0:
                self._preprocess_notes(
                    score.tracks[t],
                    tpq_resampling_factors,
                    ticks_per_beat,
                )

            # Resample pitch bend values
            if self.config.use_pitch_bends and len(score.tracks[t].pitch_bends) > 0:
                score.tracks[t].pitch_bends = self._preprocess_pitch_bends(
                    score.tracks[t].pitch_bends, tpq_resampling_factors
                )

            # Resample pedals durations
            if self.config.use_sustain_pedals and len(score.tracks[t].pedals) > 0:
                score.tracks[t].pedals = self._preprocess_pedals(
                    score.tracks[t].pedals, tpq_resampling_factors, ticks_per_beat
                )

            # Delete track only there is nothing inside being used
            if is_track_empty(
                score.tracks[t],
                check_pedals=self.config.use_sustain_pedals,
                check_pitch_bend=self.config.use_pitch_bends,
            ):
                del score.tracks[t]
                continue

        # Process tempo changes
        if self.config.use_tempos:
            score.tempos = self._preprocess_tempos(score.tempos, tpq_resampling_factors)

        # We do not change key signature changes, markers and lyrics here as they are
        # not used by MidiTok (yet)

        return score

    def _resample_score(
        self, score: Score, _new_tpq: int, _time_signatures_copy: TimeSignatureTickList
    ) -> Score:
        if score.ticks_per_quarter != _new_tpq:
            # Times of time signatures copy need to be resampled too
            time_signatures_soa = _time_signatures_copy.numpy()
            time_signatures_soa["time"] = (
                time_signatures_soa["time"] * (_new_tpq / score.ticks_per_quarter)
            ).astype(np.int32)

            score = score.resample(_new_tpq, min_dur=1)
            score.time_signatures = TimeSignature.from_numpy(
                **time_signatures_soa,
            )
        # Otherwise we do a copy in order to make sure no inplace operation is performed
        # on the provided Score object.
        # We make a copy here instead of at beginning as resample also makes a copy.
        else:
            score = score.copy()
            score.time_signatures = _time_signatures_copy

        return score

    def _filter_unsupported_time_signatures(
        self, time_signatures: TimeSignatureTickList
    ) -> None:
        """
        Remove time signatures from a list that are unsupported by the tokenizer.

        :param time_signatures: list of time signatures to filter.
        """
        for i in range(len(time_signatures) - 1, -1, -1):
            if (
                time_signatures[i].numerator,
                time_signatures[i].denominator,
            ) not in self.time_signatures:
                if self._verbose:
                    warnings.warn(
                        f"The file contains a time signature ({time_signatures[i]}) "
                        f"outside of those supported by the tokenizer ("
                        f"{self.time_signatures}). You should either discard this file"
                        f" or support this time signature, or alternatively deleting "
                        f"it however if you are using a beat-based tokenizer (REMI) "
                        f"the bars will be incorrectly detected.",
                        stacklevel=2,
                    )
                del time_signatures[i]

    def _preprocess_notes(
        self,
        track: Track,
        resampling_factors: np.ndarray = None,
        ticks_per_beat: np.ndarray = None,
        min_duration: int = 1,
    ) -> None:
        r"""
        Resample inplace the note velocities, remove notes outside of pitch range.

        Note durations will be clipped to the maximum duration that can be handled by
        the tokenizer. This is done to prevent having incorrect offset values when
        computing rests. Notes with pitches outside of self.pitch_range will be
        deleted.

        :param track: track containing the notes to resample.
        :param resampling_factors: sections of resampling factors, when we need to
            adjust the times of events to a specific ticks/beat value. This is required
            when the file has time signatures with different denominators. The factors
            are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
            per beat, and the second dimension representing the end tick of each
            section and the number of ticks per beat respectively. (default: ``None``)
        :param ticks_per_beat: array indicating the number of ticks per beat per time
            signature denominator section. The numbers of ticks per beat depend on the
            time signatures of the file being parsed. The array has a shape ``(N,2)``,
            for ``N`` changes of ticks per beat, and the second dimension representing
            the end tick of each section and the number of ticks per beat respectively.
            This argument is not required if
            ``tokenizer.config.sustain_pedal_duration`` is disabled.
            (default: ``None``)
        :param min_duration: minimum duration (in tick) to set to notes that have
            durations of 0 ticks after resampling. (default: ``1``)
        """
        note_soa = track.notes.numpy()

        # Delete notes outside of pitch range
        pitch_range = (
            self.config.drums_pitch_range
            if track.is_drum and self.config.use_pitchdrum_tokens
            else self.config.pitch_range
        )
        idx_out_of_pitch_range = np.where(
            np.logical_or(
                note_soa["pitch"] < pitch_range[0], note_soa["pitch"] > pitch_range[1]
            )
        )[0]
        if len(idx_out_of_pitch_range) > 0:
            mask = np.ones(len(note_soa["time"]), dtype=bool)
            mask[idx_out_of_pitch_range] = False
            for key in note_soa:
                note_soa[key] = note_soa[key][mask]
        if len(note_soa["time"]) == 0:
            track.notes = NoteTickList()
            return

        # Compute new velocities
        if self.config.use_velocities:
            note_soa["velocity"] = np_get_closest(
                self.velocities, np.array(note_soa["velocity"])
            )

        # Adjust times if needed
        if resampling_factors is not None:
            # First get the idx of the notes covered per section
            resampling_factors = self.__convert_resampling_ratios_ticks_to_idx(
                resampling_factors, note_soa["time"]
            )
            note_soa["time"] = self._adjust_time_to_tpb(
                note_soa["time"], resampling_factors
            )

        # Resample duration values if NoteOff, otherwise adjust to the vocab
        program = -1 if track.is_drum else track.program
        if program in self.config.use_note_duration_programs:
            if not self._note_on_off and ticks_per_beat is not None:
                self._adjust_durations(note_soa, ticks_per_beat)
            elif resampling_factors is not None:
                note_soa["duration"] = self._adjust_time_to_tpb(
                    note_soa["duration"], resampling_factors, min_duration
                )
                self._adjust_offset_spanning_across_time_sig(
                    note_soa, resampling_factors
                )

        # Symusic automatically sorts the notes by (time, duration, pitch) keys when
        # reading a music file. We hence don't need to sort the notes.
        # However, when using `NoteOn`/`NoteOff`, we can encounter note order
        # alterations with the velocity values as they are not sorted on velocities and
        # that the tokens are decoded following a FIFO logic.
        # To alleviate this, a user can sort them before calling the tokenizer.
        # We do not do it here as it is not considered a disturbing issue, and that it
        # would add a significant overhead preprocessing time. This is however done in
        # the tokenization tests of MidiTok for concerned tokenizers in order to keep
        # 100% of the data integrity, so that the tests pass.

        notes_new = Note.from_numpy(**note_soa)

        if self.config.remove_duplicated_notes:
            # we need to resort here, as symusic does it by (time, duration, pitch).
            notes_new.sort(key=lambda n: (n.time, n.pitch, n.duration, n.velocity))
            remove_duplicated_notes(notes_new)

        track.notes = notes_new

    def _preprocess_tempos(
        self,
        tempos: TempoTickList,
        resampling_factors: np.ndarray = None,
    ) -> TempoTickList:
        r"""
        Resample the tempo values of tempo change events.

        For tempo changes occurring at the same tick/time, we only keep the last one.
        Consecutive identical tempo changes will be removed if
        ``self.config.delete_equal_successive_tempo_changes`` is True.

        :param tempos: tempo changes to resample.
        :param resampling_factors: sections of resampling factors, when we need to
            adjust the times of events to a specific ticks/beat value. This is required
            when the file has time signatures with different denominators. The factors
            are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
            per beat, and the second dimension representing the end tick of each
            section and the number of ticks per beat respectively. (default: ``None``)
        """
        # If we delete the successive equal tempo changes, we need to sort them by time
        # Fortunately, sorting is already performed by symusic when loading the file.

        # Use the default tempo if there is None (shouldn't happen)
        if len(tempos) == 0:
            tempos.insert(0, Tempo(0, self.default_tempo))
            return tempos

        tempos_soa = tempos.numpy()

        # Find the closest tempos
        tempos_soa["mspq"] = np_get_closest(self._tempos_mspq, tempos_soa["mspq"])

        # Adjust times if needed
        if resampling_factors is not None:
            tempos_soa["time"] = self._adjust_time_to_tpb(
                tempos_soa["time"], resampling_factors
            )

        # Find groups of tempos at the same onset ticks, equal consecutive ones
        # Keep only last tempo change for groups with same tick
        idx_groups = np.split(
            np.arange(len(tempos_soa["time"])),
            np.where(np.diff(tempos_soa["time"]) != 0)[0] + 1,
        )
        for idx_group in reversed(idx_groups):
            if len(idx_group) > 1:
                for key in tempos_soa:
                    # We don't use a mask here as the number of idx to delete is
                    # likely to be small.
                    for idx_to_del in reversed(idx_group[:-1]):
                        tempos_soa[key] = np.delete(tempos_soa[key], idx_to_del)
        # Deduplicate successive tempo changes with same tempo value
        if self.config.delete_equal_successive_tempo_changes:
            idx_groups = np.split(
                np.arange(len(tempos_soa["time"])),
                np.where(np.diff(tempos_soa["mspq"]) != 0)[0] + 1,
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    for key in tempos_soa:
                        for idx_to_del in reversed(idx_group[1:]):
                            tempos_soa[key] = np.delete(tempos_soa[key], idx_to_del)

        tempos = Tempo.from_numpy(**tempos_soa)

        # Make sure there is at least one tempo at tick 0
        if len(tempos) > 0:
            if (
                self.config.delete_equal_successive_tempo_changes
                and tempos[0].tempo == self.default_tempo
            ):
                tempos[0].time = 0
            elif tempos[0].time != 0:
                tempos.insert(0, Tempo(0, self.default_tempo))
        else:
            tempos.insert(0, Tempo(0, self.default_tempo))

        return tempos

    def _preprocess_time_signatures(
        self, time_sigs: TimeSignatureTickList, time_division: int
    ) -> None:
        r"""
        Resamples the time signature changes.

        Time signature will be delayed to the next bar. See MIDI 1.0 Detailed
        specifications, pages 54 - 56, for more information on delayed time signature
        messages.
        If the ``delete_equal_successive_time_sig_changes`` parameter is set ``True``
        in the tokenizer's configuration, the time signatures must be sorted by time
        before calling this method. This is done by symusic when loading a file. If
        this method is called for a file created from another way, make sure they
        are sorted: ``score.time_signatures.sort()``.

        :param time_sigs: time signature changes to quantize.
        :param time_division: time division in ticks per quarter of the Score.
        """

        def are_ts_equals(ts1: TimeSignature, ts2: TimeSignature) -> bool:
            return (ts1.numerator, ts1.denominator) == (ts2.numerator, ts2.denominator)

        i = 0
        ticks_per_bar = compute_ticks_per_bar(time_sigs[0], time_division)
        previous_tick = 0  # first time signature change is always at tick 0
        while i < len(time_sigs):
            # 1. If it is identical to the previous one --> delete it
            if i >= 1 and are_ts_equals(time_sigs[i], time_sigs[i - 1]):
                del time_sigs[i]
                continue

            # 2. Delay the time of each TS to the beginning of the next bar
            # Can happen if the previous ts has been delayed to the next bar
            if i >= 1 and time_sigs[i].time < time_sigs[i - 1].time:
                time_sigs[i].time = time_sigs[i - 1].time
            bar_offset, rest = divmod(time_sigs[i].time - previous_tick, ticks_per_bar)
            # time sig doesn't happen on a new bar, we update it to the next bar
            if rest > 0:
                time_sigs[i].time = previous_tick + (bar_offset + 1) * ticks_per_bar
            # Update values
            ticks_per_bar = compute_ticks_per_bar(time_sigs[i], time_division)
            previous_tick = time_sigs[i].time

            # 3. If it is at the same tick as the previous one, we delete the previous
            if i >= 1 and time_sigs[i].time == time_sigs[i - 1].time:
                del time_sigs[i - 1]
                # Check the one previous the one deleted is not equal to the current one
                if i >= 2 and are_ts_equals(time_sigs[i - 1], time_sigs[i - 2]):
                    # If it is, we delete the current one (at i-1 now) and decrement i
                    del time_sigs[i - 1]
                    ticks_per_bar = compute_ticks_per_bar(
                        time_sigs[i - 2], time_division
                    )
                    previous_tick = time_sigs[i - 2].time
                    i -= 1
                continue

            # 4. Pass to the next one
            i += 1

        # Make sure there is at least one time signature at tick 0
        if len(time_sigs) > 0:
            if (
                self.config.delete_equal_successive_time_sig_changes
                and (time_sigs[0].numerator, time_sigs[0].denominator) == TIME_SIGNATURE
            ):
                time_sigs[0].time = 0
            elif time_sigs[0].time != 0:
                time_sigs.insert(0, TimeSignature(0, *TIME_SIGNATURE))
        else:
            time_sigs.insert(0, TimeSignature(0, *TIME_SIGNATURE))

    def _preprocess_pitch_bends(
        self,
        pitch_bends: PitchBendTickList,
        resampling_factors: np.ndarray = None,
    ) -> PitchBendTickList:
        r"""
        Resample the pitch bend events from a track.

        Overlapping pitch bends will be deduplicated by keeping the one having the
        highest absolute value at a given tick.

        :param pitch_bends: pitch bend events.
        :param resampling_factors: sections of resampling factors, when we need to
            adjust the times of events to a specific ticks/beat value. This is required
            when the Score has time signatures with different denominators. The factors
            are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
            per beat, and the second dimension representing the end tick of each
            section and the number of ticks per beat respectively. (default: ``None``)
        """
        pitch_bends_soa = pitch_bends.numpy()

        # Find closest value
        pitch_bends_soa["value"] = np_get_closest(
            self.pitch_bends, pitch_bends_soa["value"]
        )

        # Adjust times if needed
        if resampling_factors is not None:
            pitch_bends_soa["time"] = self._adjust_time_to_tpb(
                pitch_bends_soa["time"], resampling_factors
            )

        # Find groups of pitch bends at the same onset ticks, and keep the > abs values
        if len(pitch_bends) > 1:
            idx_groups = np.split(
                np.arange(len(pitch_bends_soa["value"])),
                np.where(np.diff(pitch_bends_soa["time"]) != 0)[0] + 1,
            )
            for idx_group in reversed(idx_groups):
                if len(idx_group) > 1:
                    values_group = pitch_bends_soa["value"][idx_group]
                    max_abs_idx = np.argmax(np.abs(values_group))
                    pitch_bends_soa["value"][idx_group[0]] = values_group[max_abs_idx]
                    for key in pitch_bends_soa:
                        for idx_to_del in reversed(idx_group[1:]):
                            pitch_bends_soa[key] = np.delete(
                                pitch_bends_soa[key], idx_to_del
                            )

        return PitchBend.from_numpy(**pitch_bends_soa)

    def _preprocess_pedals(
        self,
        pedals: PedalTickList,
        resampling_factors: np.ndarray = None,
        ticks_per_beat: np.ndarray = None,
        min_duration: int = 1,
    ) -> PedalTickList:
        r"""
        Resamples the pedals durations.

        :param pedals: pedals to preprocess.
        :param resampling_factors: sections of resampling factors, when we need to
            adjust the times of events to a specific ticks/beat value. This is required
            when the Score has time signatures with different denominators. The factors
            are given as a numpy array of shape ``(N,2)``, for ``N`` changes of ticks
            per beat, and the second dimension representing the end tick of each
            section and the number of ticks per beat respectively. (default: ``None``)
        :param ticks_per_beat: array indicating the number of ticks per beat per
            portions. The numbers of ticks per beat depend on the time signatures of
            the Score being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively. This argument
            is not required if ``tokenizer.config.sustain_pedal_duration`` is disabled.
            (default: ``None``)
        :param min_duration: minimum duration (in tick) to set to notes that have
            durations of 0 ticks after resampling. (default: ``1``)
        """
        pedals_soa = pedals.numpy()

        # Adjust times if needed
        if resampling_factors is not None:
            # First get the idx of the notes covered per section
            resampling_factors_ = self.__convert_resampling_ratios_ticks_to_idx(
                resampling_factors, pedals_soa["time"]
            )
            pedals_soa["time"] = self._adjust_time_to_tpb(
                pedals_soa["time"], resampling_factors_
            )

        # Format durations (if needed) and merge successive pedals
        end_arr = pedals_soa["time"] + pedals_soa["duration"]
        # Idx of the pedals for which the end time is higher than the time of the next
        overlapping_mask = end_arr[:-1] >= pedals_soa["time"][1:]
        idx_overlap = np.where(overlapping_mask)[0]
        if len(idx_overlap) > 0:
            # Get the first idx of each group of successive values == True
            idx_diff = np.diff(idx_overlap, prepend=-2)  # prep to keep the first idx
            first_idx_groups = idx_overlap[idx_diff > 1]

            # Merge PedalOn periods depending on their durations
            idx_to_delete = []
            for idx in first_idx_groups:
                ip = 1
                while (
                    idx + ip < len(pedals_soa["time"])
                    and end_arr[idx] >= pedals_soa["time"][idx + ip]
                ):
                    pedals_soa["duration"][idx] = max(
                        pedals_soa["duration"][idx],
                        end_arr[idx + ip] - pedals_soa["time"][idx],
                    )
                    end_arr[idx] = pedals_soa["time"][idx] + pedals_soa["duration"][idx]
                    idx_to_delete.append(idx + ip)
                    ip += 1

            # Deduplicate pedals
            mask = np.ones(len(pedals_soa["time"]), dtype=bool)
            idx_to_delete = np.array(idx_to_delete)
            mask[idx_to_delete] = False
            for key in pedals_soa:
                pedals_soa[key] = pedals_soa[key][mask]

        # Resample duration values if NoteOff, otherwise adjust to the vocab
        if self.config.sustain_pedal_duration and ticks_per_beat is not None:
            self._adjust_durations(pedals_soa, ticks_per_beat)
        elif resampling_factors is not None:
            resampling_factors_ = self.__convert_resampling_ratios_ticks_to_idx(
                resampling_factors, pedals_soa["time"]
            )
            pedals_soa["duration"] = self._adjust_time_to_tpb(
                pedals_soa["duration"], resampling_factors_, min_duration
            )
            self._adjust_offset_spanning_across_time_sig(
                pedals_soa, resampling_factors_
            )

        return Pedal.from_numpy(**pedals_soa)

    @staticmethod
    def _adjust_time_to_tpb(
        times_arr: np.ndarray,
        tpq_resampling_factor: np.ndarray,
        min_duration: int | None = None,
    ) -> np.ndarray:
        # Batch by factor (i.e. time signature denominator) section
        idx_first = 0
        for rf_idx, (idx_last, factor) in enumerate(tpq_resampling_factor):
            idx_last_ = None if rf_idx == len(tpq_resampling_factor) - 1 else idx_last
            # Round time values to the factor
            # Except if the factor is 1, it means that the tpb is equal to the tpq
            if factor != 1:
                times_arr[idx_first:idx_last_] = (
                    np.round(times_arr[idx_first:idx_last_] / factor) * factor
                )
                if min_duration is not None:
                    times_arr[idx_first:idx_last_][
                        times_arr[idx_first:idx_last_] == 0
                    ] = min_duration * factor
            idx_first = idx_last_

        return times_arr

    @staticmethod
    def _adjust_offset_spanning_across_time_sig(
        notes_pedals_soa: dict[str, np.ndarray],
        resampling_factors: np.ndarray,
    ) -> None:
        end_arr = notes_pedals_soa["time"] + notes_pedals_soa["duration"]
        idx_first = 0
        for idx_fact, (idx_last, _) in enumerate(resampling_factors[:-1]):
            # NoteOff/PedalOff idx with durations spanning across time sigs adjust end
            spanning_durations_idx = np.where(
                end_arr[idx_first:idx_last] >= notes_pedals_soa["time"][idx_last]
            )[0]
            for idx in spanning_durations_idx:
                # Get the factor for the idx as it can be different from the next one
                factor_for_idx = resampling_factors[
                    np.argmax(
                        resampling_factors[idx_fact:, 0]
                        >= notes_pedals_soa["time"][idx]
                    ),
                    1,
                ]
                new_end = np.round(end_arr[idx] / factor_for_idx) * factor_for_idx
                notes_pedals_soa["duration"][idx] = (
                    new_end - notes_pedals_soa["time"][idx]
                )
            idx_first = idx_last

    def _adjust_durations(
        self, notes_pedals_soa: dict[str, np.ndarray], ticks_per_beat: np.ndarray
    ) -> None:
        """
        Adjust the durations of notes or pedals.

        The durations of notes or pedals will be set to the closest ones of those in
        the tokenizer's vocabulary. The new durations are calculated depending on the
        time signature, i.e. number of ticks in a beat.

        :param notes_pedals_soa: structure of arrays (soa) of notes or pedals.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            portions. The numbers of ticks per beat depend on the time signatures of
            the Score being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
        """
        # Batch by tpb section
        dur_idx_first = 0
        tick_last_event = notes_pedals_soa["time"][-1]
        for tpb_idx, (last_tick_tpb, tpb) in enumerate(ticks_per_beat):
            # Get idx of the concerned notes.
            # There shouldn't be equal successive tpb values in ticks_per_beat.
            # If last tpb --> set last note to max_tick to avoid iterating notes
            if tpb_idx + 1 == len(ticks_per_beat) or last_tick_tpb > tick_last_event:
                dur_idx_last = None
            else:
                dur_idx_last = dur_idx_first + np.argmax(
                    notes_pedals_soa["time"][dur_idx_first:] >= last_tick_tpb
                )
            notes_pedals_soa["duration"][dur_idx_first:dur_idx_last] = np_get_closest(
                self._tpb_to_time_array[tpb],
                notes_pedals_soa["duration"][dur_idx_first:dur_idx_last],
            )
            dur_idx_first = dur_idx_last
            if dur_idx_last is None:
                break

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence | list[TokSequence]:
        r"""
        Convert a **preprocessed** ``symusic.Score`` object to a sequence of tokens.

        The workflow of this method is as follows: the global events (*Tempo*,
        *TimeSignature*...) and track events (*Pitch*, *Velocity*, *Pedal*...) are
        gathered into a list, then the time events are added. If ``one_token_stream`` is
        ``True``, all events of all tracks are treated all at once, otherwise the
        events of each track are treated independently.

        :param score: the :class:`symusic.Score` object to convert.
        :param attribute_controls_indexes: indices of the attribute controls to compute
            and associated tracks and bars. This argument has to be provided as a
            dictionary mapping track indices to dictionaries mapping attribute control
            indices (indexing ``tokenizer.attribute_controls``) to a sequence of bar
            indexes if the AC is "bar-level" or anything if it is "track-level".
            Its structure is as:
            ``{track_idx: {ac_idx: Any (track ac) | [bar_idx, ...] (bar ac)}}``
            This argument is meant to be used when training a model in order to make it
            learn to generate tokens accordingly to the attribute controls.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        # Create events list
        all_events = []
        if not self.config.one_token_stream_for_programs:
            if len(score.tracks) == 0:
                all_events.append([])
            else:
                all_events = [[] for _ in range(len(score.tracks))]
        if attribute_controls_indexes is None:
            attribute_controls_indexes = {}

        # Global events (Tempo, TimeSignature)
        global_events = self._create_global_events(score)
        if self.config.one_token_stream_for_programs:
            all_events += global_events
        else:
            for i in range(len(all_events)):
                all_events[i] += global_events

        # Compute ticks_per_beat sections depending on the time signatures
        # This has to be computed several times, in preprocess after resampling & here.
        if (
            not self._note_on_off
            or (self.config.use_sustain_pedals and self.config.sustain_pedal_duration)
            or self.config.use_chords
            or self.config.use_pitch_intervals
        ):
            if self.config.use_time_signatures and len(score.time_signatures) > 0:
                ticks_per_beat = get_score_ticks_per_beat(score)
            else:
                ticks_per_beat = np.array([[score.end(), self.time_division]])
        else:
            ticks_per_beat = None

        # Adds track tokens
        ticks_bars = get_bars_ticks(score, only_notes_onsets=True)
        ticks_beats = get_beats_ticks(score, only_notes_onsets=True)
        for ti, track in enumerate(score.tracks):
            track_events = self._create_track_events(
                track,
                ticks_per_beat,
                score.ticks_per_quarter,
                ticks_bars,
                ticks_beats,
                attribute_controls_indexes.get(ti, None),
            )
            if self.config.one_token_stream_for_programs:
                all_events += track_events
            else:
                all_events[ti] += track_events
                self._sort_events(all_events[ti])
        if self.config.one_token_stream_for_programs:
            self._sort_events(all_events)
            # Add ProgramChange (named Program) tokens if requested.
            if self.config.program_changes:
                self._insert_program_change_events(all_events)
        # Special case where there are only tempos/time sigs, we still need to sort them
        elif len(score.tracks) == 0 and len(all_events[0]) > 2:
            self._sort_events(all_events[0])

        # Add time events
        if self.config.one_token_stream_for_programs:
            all_events = self._add_time_events(all_events, score.ticks_per_quarter)
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                all_events[i] = self._add_time_events(
                    all_events[i], score.ticks_per_quarter
                )
                # Add program tokens at the beginning of the sequences if using
                # "program_changes" and not in "one_token_stream" mode.
                if self.config.program_changes and len(score.tracks) > 0:
                    program = (
                        score.tracks[i].program if not score.tracks[i].is_drum else -1
                    )
                    # If the first token is a "Track_Start" the program token is
                    # inserted after
                    all_events[i].insert(
                        0 if all_events[i][0].type_ != "Track" else 1,
                        Event("Program", program, 0),
                    )
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

    def _sort_events(self, events: list[Event]) -> None:
        # Can be overridden by subclasses if required (MIDILike)
        events.sort(key=lambda e: e.time)
        # Set Events of track-level attribute controls from -1 to 0 after sorting
        if len(self.attribute_controls) > 0:
            for event in events:
                if not event.type_.startswith("ACTrack"):
                    break
                event.time = 0

    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: np.ndarray,
        time_division: int,
        ticks_bars: Sequence[int],
        ticks_beats: Sequence[int],
        attribute_controls_indexes: Mapping[int, Sequence[int] | bool] | None = None,
    ) -> list[Event]:
        r"""
        Extract the tokens/events from a track (``symusic.Track``).

        Concerned events are: *Pitch*, *Velocity*, *Duration*, *NoteOn*, *NoteOff* and
        optionally *Chord*, *Pedal* and *PitchBend*.
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in**
        :py:func:`miditok.MusicTokenizer.preprocess_score`.

        :param track: ``symusic.Track`` to extract events from.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            section. The numbers of ticks per beat depend on the time signatures of
            the Score being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
            This argument is not required if the tokenizer is not using *Duration*,
            *PitchInterval* or *Chord* tokens. (default: ``None``)
        :param time_division: time division in ticks per quarter note of the file.
        :param ticks_bars: ticks indicating the beginning of each bar.
        :param ticks_beats: ticks indicating the beginning of each beat.
        :param attribute_controls_indexes: indices of the attribute controls to compute
            This argument has to be provided as a dictionary mapping attribute control
            indices (indexing ``tokenizer.attribute_controls``) to a sequence of
            bar indexes if the AC is "bar-level" or anything if it is "track-level".
            Its structure is as: ``{ac_idx: Any (track ac) | [bar_idx, ...] (bar ac)}``
            This argument is meant to be used when training a model in order to make it
            learn to generate tokens accordingly to the attribute controls.
        :return: sequence of corresponding ``Event``s.
        """
        program = track.program if not track.is_drum else -1
        use_durations = program in self.config.use_note_duration_programs
        events = []
        # max_time_interval is adjusted depending on the time signature denom / tpb
        max_time_interval = 0
        if self.config.use_pitch_intervals:
            max_time_interval = round(
                (
                    ticks_per_beat[0, 1] * self.config.pitch_intervals_max_time_dist
                ).item()
            )
        previous_note_onset = -max_time_interval - 1
        previous_pitch_onset = -128  # lowest at a given time
        previous_pitch_chord = -128  # for chord intervals

        # Attribute controls
        if attribute_controls_indexes:
            for ac_idx, tracks_bars_idx in attribute_controls_indexes.items():
                if (
                    isinstance(self.attribute_controls[ac_idx], BarAttributeControl)
                    and len(tracks_bars_idx) == 0
                ):
                    continue
                events += self.attribute_controls[ac_idx].compute(
                    track,
                    time_division,
                    ticks_bars,
                    ticks_beats,
                    tracks_bars_idx,
                )

        # Add sustain pedal
        if self.config.use_sustain_pedals:
            tpb_idx = 0
            for pedal in track.pedals:
                # If not using programs, the default value is 0
                events.append(
                    Event(
                        "Pedal",
                        program,
                        pedal.time,
                        program,
                    )
                )
                # PedalOff or Duration
                if self.config.sustain_pedal_duration:
                    # `while` here as there might not be any note in the next section
                    while pedal.time >= ticks_per_beat[tpb_idx, 0]:
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

        # Creates the Note On, Note Off and Velocity events
        tpb_idx = 0
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
                    max_time_interval = round(
                        (
                            ticks_per_beat[tpb_idx, 1]
                            * self.config.pitch_intervals_max_time_dist
                        ).item()
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
                if self.config.use_pitchdrum_tokens and track.is_drum:
                    note_token_name = "DrumOn" if self._note_on_off else "PitchDrum"
                else:
                    note_token_name = "NoteOn" if self._note_on_off else "Pitch"
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
            if self.config.use_velocities:
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
            if use_durations:
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
                            type_="DrumOff"
                            if self.config.use_pitchdrum_tokens and track.is_drum
                            else "NoteOff",
                            value=note.pitch,
                            time=note.end,
                            program=program,
                            desc=note.end,
                        )
                    )
                else:
                    events.append(
                        self._create_duration_event(
                            note=note,
                            _program=program,
                            _ticks_per_beat=ticks_per_beat,
                            _tpb_idx=tpb_idx,
                        )
                    )

        return events

    def _create_duration_event(
        self, note: Note, _program: int, _ticks_per_beat: np.ndarray, _tpb_idx: int
    ) -> Event:
        while note.time >= _ticks_per_beat[_tpb_idx, 0]:
            _tpb_idx += 1
        dur = self._tpb_ticks_to_tokens[_ticks_per_beat[_tpb_idx, 1]][note.duration]
        return Event(
            type_="Duration",
            value=dur,
            time=note.start,
            program=_program,
            desc=f"{note.duration} ticks",
        )

    @staticmethod
    def _insert_program_change_events(events: list[Event]) -> None:
        """
        Add inplace *Program* tokens acting as Program Changes to a list of ``Event``s.

        :param events: Events to add Programs
        """
        previous_program = None
        previous_type = None
        program_change_events = []
        for ei, event in enumerate(events):
            if (
                event.program is not None
                and event.program != previous_program
                and event.type_ not in ["Pedal", "PedalOff", *TOKEN_TYPE_BEFORE_PC]
                and not (event.type_ == "Duration" and previous_type == "Pedal")
            ):
                previous_program = event.program
                program_change_events.append(
                    (ei, Event("Program", event.program, event.time))
                )
            previous_type = event.type_

        for idx, event in reversed(program_change_events):
            events.insert(idx, event)

    def _create_global_events(self, score: Score) -> list[Event]:
        r"""
        Create the *global* music tokens: ``Tempo`` and ``TimeSignature``.

        :param score: ``symusic.Score`` to extract the events from.
        :return: list of ``miditok.classes.Event``.
        """
        events = []

        # First adds time signature tokens if specified
        if self.config.use_time_signatures:
            events += [
                Event(
                    type_="TimeSig",
                    value=f"{time_sig.numerator}/{time_sig.denominator}",
                    time=time_sig.time,
                )
                for time_sig in score.time_signatures
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
                for tempo in score.tempos
            ]

        return events

    @abstractmethod
    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """
        raise NotImplementedError

    def midi_to_tokens(self, *args, **kwargs) -> Score | list[Score]:  # noqa: D102, ANN002
        warnings.warn(
            "miditok: The `midi_to_tokens` method had been renamed `encode`. It is now "
            "depreciated and will be removed in future updates.",
            stacklevel=2,
        )
        return self.encode(*args, **kwargs)

    def encode(
        self,
        score: Score | Path,
        encode_ids: bool = True,
        no_preprocess_score: bool = False,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence | list[TokSequence]:
        r"""
        Tokenize a music file (MIDI/abc), given as a ``symusic.Score`` or a file path.

        You can provide a ``Path`` to the file to tokenize, or a ``symusic.Score``
        object.
        This method returns a (list of) :class:`miditok.TokSequence`.

        If you are implementing your own tokenization by subclassing this class,
        **override the protected** ``_score_to_tokens`` **method**.

        :param score: the ``symusic.Score`` object to convert.
        :param encode_ids: the backbone model (BPE, Unigram, WordPiece) will encode the
            tokens and compress the sequence. Can only be used if the tokenizer has been
            trained. (default: ``True``)
        :param no_preprocess_score: whether to preprocess the ``symusic.Score``. If this
            argument is provided as ``True``, make sure that the corresponding music
            file / ``symusic.Score`` has already been preprocessed by the tokenizer
            (:py:func:`miditok.MusicTokenizer.preprocess_score`) or that its content is
            aligned with the tokenizer's vocabulary, otherwise the tokenization is
            likely to crash. This argument is useful in cases where you need to use the
            preprocessed ``symusic.Score`` along with the tokens to not have to
            preprocess it twice as this method preprocesses it inplace.
            (default: ``False``)
        :param attribute_controls_indexes: indices of the attribute controls to compute
            and associated tracks and bars. This argument has to be provided as a
            dictionary mapping track indices to dictionaries mapping attribute control
            indices (indexing ``tokenizer.attribute_controls``) to a sequence of bar
            indexes if the AC is "bar-level" or anything if it is "track-level".
            Its structure is as:
            ``{track_idx: {ac_idx: Any (track ac) | [bar_idx, ...] (bar ac)}}``
            This argument is meant to be used when training a model in order to make it
            learn to generate tokens accordingly to the attribute controls. For maximum
            safety, it should be used with ``no_preprocess_score`` with an already
            preprocessed ``symusic.Score`` in order to make sure that the provided
            tracks indexes will remain correct as the preprocessing might delete or
            merge tracks depending on the tokenizer's configuration.
        :return: a :class:`miditok.TokSequence` if ``tokenizer.one_token_stream`` is
            ``True``, else a list of :class:`miditok.TokSequence` objects.
        """
        # Load the file if a path was given
        if not isinstance(score, ScoreTick):
            score = Score(score)

        # Preprocess the music file
        if not no_preprocess_score:
            score = self.preprocess_score(score)

        # Tokenize it
        tokens = self._score_to_tokens(score, attribute_controls_indexes)
        # Add bar/beat ticks here to TokSeq as they need to be from preprocessed Score
        add_bar_beats_ticks_to_tokseq(tokens, score)

        # Encode the ids if the tokenizer is trained
        if encode_ids and self.is_trained:
            self.encode_token_ids(tokens)

        return tokens

    def complete_sequence(self, seq: TokSequence, complete_bytes: bool = False) -> None:
        r"""
        Complete (inplace) a :class:`miditok.TokSequence`.

        The input sequence can have some of its attributes (``ids``, ``tokens``) not
        initialized (i.e. ``None``). This method will initialize them from the present
        ones. The ``events`` attribute will not be filled as it is only intended for
        debug purpose. The ``bytes`` attribute will be created if ``complete_bytes`` is
        provided as ``True`` and if the tokenizer has been trained.

        :param seq: input :class:`miditok.TokSequence`, must have at least one
            attribute defined.
        :param complete_bytes: will complete the bytes form of each token. This is only
            applicable if the tokenizer has been trained.
        """
        if len(seq.tokens) == 0:
            if len(seq.events) > 0:
                seq.tokens = self._events_to_tokens(seq.events)
            elif len(seq.ids) > 0:
                seq.tokens = self._ids_to_tokens(seq.ids)
            elif len(seq.bytes) > 0:
                seq.tokens = self._bytes_to_tokens(seq.bytes)
        if len(seq.ids) == 0:
            seq.ids = self._tokens_to_ids(seq.tokens)

        if complete_bytes and self.is_trained and len(seq.bytes) == 0:
            seq.bytes = self._ids_to_bytes(seq.ids, as_one_str=True)

    def _tokens_to_ids(
        self, tokens: Sequence[str | list[str]]
    ) -> list[int | list[int]]:
        r"""
        Convert a list of tokens (str) into their ids format (int).

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
        r"""
        Convert a sequence of ids (int) to their tokens format (str or Event).

        **This method will not work with ids encoded with the tokenizer's model. You
        will need to decode them first (
        :py:meth:`miditok.MusicTokenizer.decode_token_ids`)**.

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
        r"""
        Convert a sequence of ``Events`` to their tokens format (str).

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
        r"""
        Convert a list of ids into their bytes format.

        It can be returned either as a list of bytes or as a unique string of bytes.
        **This method will not work with ids encoded with the tokenizer's model. You
        will need to decode them first (
        :py:meth:`miditok.MusicTokenizer.decode_token_ids`)**.

        :param ids: token ids (int) to convert.
        :param as_one_str: will return the bytes all concatenated into one string.
            (default: ``False``)
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
        r"""
        Convert a sequence of bytes into their associated tokens (str or ``Event``).

        :param bytes_: sequence of bytes to convert.
        :param as_str: return the events as string objects, otherwise ``Event`` objects
            (default: ``True``)
        :return: the sequence of corresponding tokens (str).
        """
        if len(bytes_) == 0:
            return []
        if isinstance(bytes_[0], list):  # multiple vocabularies
            return [self._bytes_to_tokens(byte_) for byte_ in bytes_]

        tokens = []
        for byte_ in bytes_:
            token_str = self._vocab_learned_bytes_to_tokens[byte_]
            tokens.append(token_str if as_str else Event(*token_str.split("_")))
        return [tok for toks in tokens for tok in toks]  # flatten

    def _convert_sequence_to_tokseq(
        self,
        input_seq: list[int | str | list[int | str]] | np.ndarray,
    ) -> TokSequence | list[TokSequence]:
        r"""
        Convert a sequence of tokens/ids into a (list of) :class:`miditok.TokSequence`.

        The method automatically format the sequence following the tokenizer's i/o
        format.

        :param input_seq: sequence to convert. It can be a list of ids (integers),
            tokens (string) or events (Event). It can also be a Pytorch or TensorFlow
            tensor, or Numpy array representing ids.
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
            msg = (
                f"The input sequence does not have the expected dimension "
                f"({num_seq_dims} instead of {num_io_dims})."
            )
            raise ValueError(msg)

        # Convert to TokSequence
        if not self.one_token_stream and num_io_dims == num_seq_dims:
            seq = []
            for obj in arg[1]:
                kwarg = {arg[0]: obj}
                seq.append(TokSequence(**kwarg))
                if self.is_trained and seq[-1].ids is not None:
                    seq[-1].are_ids_encoded = self._are_ids_encoded(seq[-1].ids)
        else:  # 1 subscript, one_token_stream and no multi-voc
            kwarg = {arg[0]: arg[1]}
            seq = TokSequence(**kwarg)
            if self.is_trained:
                seq.are_ids_encoded = self._are_ids_encoded(seq.ids)

        return seq

    def _are_ids_encoded(self, ids: list[int] | np.ndarray) -> bool:
        r"""
        Indicate if token ids are encoded with a model (BPE, Unigram, WordPiece).

        This is performed by checking if any id has a value superior or equal to the
        length of the base vocabulary.

        :param ids: ids to check.
        :return: boolean, ``True`` if ids are encoded by a model, ``False`` otherwise.
        """
        return np.any(np.array(ids) >= len(self.vocab))

    def _preprocess_tokseq_before_decoding(self, tokseq: TokSequence) -> None:
        if len(tokseq.tokens) == 0:
            if tokseq.are_ids_encoded:
                self.decode_token_ids(tokseq)
            self.complete_sequence(tokseq)

    def tokens_to_midi(self, *args, **kwargs) -> Score:  # noqa: D102, ANN002
        warnings.warn(
            "miditok: The `tokens_to_midi` method had been renamed `decode`. It is now "
            "depreciated and will be removed in future updates.",
            stacklevel=2,
        )
        return self.decode(*args, **kwargs)

    def decode(
        self,
        tokens: TokSequence | list[TokSequence] | list[int | list[int]] | np.ndarray,
        programs: list[tuple[int, bool]] | None = None,
        output_path: str | Path | None = None,
    ) -> Score:
        r"""
        Detokenize one or several sequences of tokens into a ``symusic.Score``.

        You can give the tokens sequences either as :class:`miditok.TokSequence`
        objects, lists of integers, numpy arrays or PyTorch/Jax/Tensorflow tensors.
        The Score's time division will be the same as the tokenizer's:
        ``tokenizer.time_division``.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence`, a Tensor (PyTorch and Tensorflow are
            supported), a numpy array or a Python list of ints. The first dimension
            represents tracks, unless the tokenizer handle tracks altogether as a
            single token sequence (``tokenizer.one_token_stream == True``).
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :param output_path: path to save the file. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        if not isinstance(tokens, (TokSequence, list)) or (
            isinstance(tokens, list)
            and any(not isinstance(seq, TokSequence) for seq in tokens)
        ):
            tokens = self._convert_sequence_to_tokseq(tokens)

        # Preprocess TokSequence(s)
        if isinstance(tokens, TokSequence):
            self._preprocess_tokseq_before_decoding(tokens)
        else:  # list[TokSequence]
            for seq in tokens:
                self._preprocess_tokseq_before_decoding(seq)

        score = self._tokens_to_score(tokens, programs)

        # Create controls for pedals
        # This is required so that they are saved when the Score is dumped, as symusic
        # will only write the control messages.
        if self.config.use_sustain_pedals:
            for track in score.tracks:
                for pedal in track.pedals:
                    track.controls.append(ControlChange(pedal.time, 64, 127))
                    track.controls.append(ControlChange(pedal.end, 64, 0))
                if len(track.pedals) > 0:
                    track.controls.sort()

        # Set default tempo and time signatures at tick 0 if not present
        if len(score.tempos) == 0 or score.tempos[0].time != 0:
            score.tempos.insert(0, Tempo(0, self.default_tempo))
        if len(score.time_signatures) == 0 or score.time_signatures[0].time != 0:
            score.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))

        # Write file
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            if output_path.suffix in ABC_FILES_EXTENSIONS:
                score.dump_abc(output_path)
            else:
                score.dump_midi(output_path)
        return score

    @abstractmethod
    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        raise NotImplementedError

    def _disable_attribute_controls(self) -> None:
        # To be called in `_tweak_config_before_creating_voc` by tokenizers classes
        self.config.ac_polyphony_track = False
        self.config.ac_polyphony_bar = False
        self.config.ac_pitch_class_bar = False
        self.config.ac_note_density_track = False
        self.config.ac_note_density_bar = False
        self.config.ac_note_duration_bar = False
        self.config.ac_note_duration_track = False
        self.config.ac_repetition_track = False

    @abstractmethod
    def _create_base_vocabulary(self) -> list[str | list[str]]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        **Attribute control tokens are added when creating the tokenizer by the**
        ``MusicTokenizer.add_attribute_control`` **method.**

        :return: the vocabulary as a list of string.
        """
        raise NotImplementedError

    def __create_vocabulary(self) -> None:
        r"""
        Create the vocabulary of the tokenizer as a dictionary.

        This method is called during the tokenizer's initialization, and requires
        ``_create_vocabulary`` to be implemented by a child class.
        """
        vocab = self._create_base_vocabulary()

        if isinstance(vocab[0], list):  # multi-voc
            self._vocab_base = [{} for _ in range(len(vocab))]
            self.__vocab_base_inv = [{} for _ in range(len(vocab))]
            for vid in range(len(vocab)):
                vocab[vid] = self.special_tokens + vocab[vid]
                for tok in vocab[vid]:
                    self.add_to_vocab(tok, vocab_idx=vid)
        else:
            for tok in self.special_tokens:
                self.add_to_vocab(tok, special_token=True)
            for tok in vocab:
                self.add_to_vocab(tok)

    def _add_note_tokens_to_vocab_list(self, vocab: list[str]) -> None:
        # NoteOn + NoteOff
        pitch_values = list(
            range(self.config.pitch_range[0], self.config.pitch_range[1] + 1)
        )
        if self._note_on_off:
            vocab += [f"NoteOn_{i}" for i in pitch_values]
            if len(self.config.use_note_duration_programs) > 0:
                vocab += [f"NoteOff_{i}" for i in pitch_values]
        # Pitch + Duration (later done after velocity)
        else:
            vocab += [f"Pitch_{i}" for i in pitch_values]

        # Velocity
        if self.config.use_velocities:
            vocab += [f"Velocity_{i}" for i in self.velocities]

        # Duration
        if (
            not self._note_on_off and self.config.using_note_duration_tokens
        ) or self.config.sustain_pedal_duration:
            vocab += [
                f"Duration_{'.'.join(map(str, duration))}"
                for duration in self.durations
            ]

    def _add_additional_tokens_to_vocab_list(self, vocab: list[str]) -> None:
        # PitchInterval
        if self.config.use_pitch_intervals:
            for interval_type in ("PitchIntervalTime", "PitchIntervalChord"):
                vocab += [
                    f"{interval_type}_{pitch}"
                    for pitch in range(
                        -self.config.max_pitch_interval,
                        self.config.max_pitch_interval + 1,
                    )
                ]

        # PitchDrum
        if self.config.use_pitchdrum_tokens:
            drum_pitch_values = list(
                range(
                    self.config.drums_pitch_range[0],
                    self.config.drums_pitch_range[1] + 1,
                )
            )
            pitch_token_name = "DrumOn" if self._note_on_off else "PitchDrum"
            vocab += [f"{pitch_token_name}_{pitch}" for pitch in drum_pitch_values]
            if self._note_on_off:
                vocab += [f"DrumOff_{pitch}" for pitch in drum_pitch_values]

        # Chord
        if self.config.use_chords:
            vocab += self._create_chords_tokens()

        # Rest
        if self.config.use_rests:
            vocab += [f"Rest_{'.'.join(map(str, rest))}" for rest in self.rests]

        # Tempo
        if self.config.use_tempos:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        # Program
        if self.config.use_programs:
            vocab += [f"Program_{program}" for program in self.config.programs]

        # TimeSig
        if self.config.use_time_signatures:
            vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]

        # Pedal
        if self.config.use_sustain_pedals:
            if self.config.use_programs:
                vocab += [f"Pedal_{program}" for program in self.config.programs]
                if not self.config.sustain_pedal_duration:
                    vocab += [f"PedalOff_{program}" for program in self.config.programs]
            else:
                vocab.append("Pedal_0")
                if not self.config.sustain_pedal_duration:
                    vocab.append("PedalOff_0")

        # PitchBend
        if self.config.use_pitch_bends:
            vocab += [f"PitchBend_{pitch_bend}" for pitch_bend in self.pitch_bends]

    def _update_token_types_indexes(self) -> None:
        r"""Update the _token_types_indexes attribute according to _event_to_token."""

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
        r"""
        Return the list of token ids of the given type.

        :param token_type: token type to get the associated token ids.
        :param vocab_id: index of the vocabulary associated to the token, if applicable.
            (default: ``None``)
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
        special_token: bool = False,
        vocab_idx: int | None = None,
        byte_: str | None = None,
    ) -> None:
        r"""
        Add an event to the vocabulary. Its id will be the length of the vocab.

        :param token: token to add, as a formatted string of the form "Type_Value",
            e.g. Pitch_80, or an Event.
        :param special_token: whether the token is special. (default: ``False``)
        :param vocab_idx: idx of the vocabulary (in case of embedding pooling).
            (default: ``None``)
        :param byte_: unique byte associated to the token. The associated byte of a
            token is used to encode-decode ids with the tokenizer's model (BPE, Unigram,
            WordPiece). If None is given, it will default to ``chr(id_ + CHR_ID_START)``
            . (default: ``None``)
        """
        token_str = token if isinstance(token, str) else str(token)

        if special_token:
            parts = token_str.split("_")
            if len(parts) == 1:
                parts.append("None")
            elif len(parts) > 2:
                parts = ["-".join(parts[:-1]), parts[-1]]
            token = "_".join(parts)
            if token not in self.config.special_tokens:
                self.config.special_tokens.append(token)

        dict_vocab = self.vocab if vocab_idx is None else self.vocab[vocab_idx]
        if token_str in dict_vocab:
            token_id = dict_vocab[token_str]
            warnings.warn(
                f"Token {token_str} is already in the vocabulary at idx {token_id}.",
                stacklevel=2,
            )
        elif vocab_idx is not None:
            self._vocab_base[vocab_idx][token_str] = len(self._vocab_base[vocab_idx])
            self.__vocab_base_inv[vocab_idx][len(self.__vocab_base_inv[vocab_idx])] = (
                token_str
            )
        else:
            id_ = len(self._model.get_vocab()) if self.is_trained else len(self.vocab)
            self._vocab_base[token_str] = id_
            self.__vocab_base_inv[len(self.__vocab_base_inv)] = token_str

            # Byte
            if byte_ is None:
                byte_ = chr(id_ + CHR_ID_START)
            self._vocab_base_id_to_byte[id_] = byte_
            self._vocab_base_byte_to_token[byte_] = token
            if self._model is not None:
                self._model.add_tokens([byte_])

    def _create_chords_tokens(self) -> list[str]:
        """
        Create the *Chord* tokens that will populate the base vocabulary.

        This protected method is intended to be used when creating the vocabulary.

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
        r"""
        Return the type of the given token id.

        :param id_: token id to get the type.
        :param vocab_id: index of the vocabulary associated to the token, if
            applicable. (default: ``None``)
        :return: the type of the token, as a string
        """
        token = self.__get_from_voc(id_, vocab_id)
        return token.split("_")[0]

    @abstractmethod
    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Create a dictionary describing the possible token type successions.

        This method is unimplemented and need to be overridden by inheriting classes.
        See other classes (:class:`miditok.REMI._create_token_types_graph`, ...)
        for examples of how to implement it.

        :return: the token types transitions dictionary.
        """
        raise NotImplementedError

    def _add_special_tokens_to_types_graph(self) -> None:
        r"""
        Add (inplace) special tokens types to the token types graph dictionary.

        Two exceptions are made for the special BOS (Beginning of Sequence) and EOS
        (End of Sequence) tokens: No token type can precede a BOS token, and EOS token
        cannot precede any other token.
        """
        original_token_types = set(self.tokens_types_graph.keys())
        for special_token in self.config.special_tokens:
            special_token_type = special_token.split("_")[0]
            if special_token_type == EOS_TOKEN_NAME:
                self.tokens_types_graph[EOS_TOKEN_NAME] = set()
            else:
                self.tokens_types_graph[special_token_type] = (
                    original_token_types | set(self.config.special_tokens)
                )

            if special_token_type != BOS_TOKEN_NAME:
                for token_type in original_token_types:
                    self.tokens_types_graph[token_type].add(special_token_type)

    def _create_durations_tuples(self) -> list[tuple[int, int, int]]:
        r"""
        Create the possible durations in beat / position units as tuples of intergers.

        The tuples follow the form: ``(beat, pos, res)`` where ``beat`` is the number
        of beats, ``pos`` the number of "positions" and ``res`` the beat resolution
        considered (positions per beat).
        Example: ``(2, 5, 8)`` means the duration is 2 beat long + position 5 / 8 of
        the ongoing beat. This would give in ticks:
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

    def __create_tpb_per_ts(self) -> dict[int, int]:
        """
        Return the dictionary of the possible ticks per beat values per time signature.

        The set of possible values depends on the tokenizer's maximum number of
        positions per beat (`self.config.max_num_pos_per_beat`) and the time signatures
        it supports.

        :return: dictionary of the possible ticks per beat values per time signature,
            keys are time signature denominators, values the ticks/beat values.
        """
        max_denom = max(ts[1] for ts in self.time_signatures)
        return {
            denom: self.config.max_num_pos_per_beat * (max_denom // denom)
            for denom in self.config.time_signature_range
        }

    def _get_score_resampling_factor(self, score: Score) -> np.ndarray:
        """
        Compute the portions of numbers of ticks in a beat in a ``symusic.Score``.

        The method returns a numpy array of shape ``(N,2)``, for N ticks-per-beat
        changes, and the second dimension corresponding to the ending tick and the
        number of ticks per beat of the portion.
        **The time signatures must be sorted by time.**

        :param score: ``symusic.Score`` to analyze.
        :return: ticks per beat values as a numpy array.
        """
        resampling_factors = [
            [
                score.time_signatures[tsi + 1].time,
                score.ticks_per_quarter
                // (
                    self.config.max_num_pos_per_beat
                    * (score.time_signatures[tsi].denominator / 4)
                ),
            ]
            for tsi in range(len(score.time_signatures) - 1)
        ]

        # Handles the last one up to the max tick of the Score
        resampling_factors.append(
            [
                score.end() + 1,
                score.ticks_per_quarter
                // (
                    self.config.max_num_pos_per_beat
                    * (score.time_signatures[-1].denominator / 4)
                ),
            ]
        )

        # Remove equal successive ones
        for i in range(len(resampling_factors) - 1, 0, -1):
            if resampling_factors[i][1] == resampling_factors[i - 1][1]:
                resampling_factors[i - 1][0] = resampling_factors[i][0]
                del resampling_factors[i]

        return np.array(resampling_factors, dtype=np.intc)

    @staticmethod
    def __convert_resampling_ratios_ticks_to_idx(
        resampling_factors: np.ndarray, time_arr: np.array
    ) -> np.ndarray:
        idx_first = 0
        factors_idx = resampling_factors.copy()
        for rf_idx, last_tick_factor in enumerate(resampling_factors):
            # Get the last concerned idx for this section.
            if rf_idx + 1 == len(resampling_factors):
                idx_last = len(time_arr) - 1
            else:
                idx_last = np.argmax(time_arr[idx_first:] >= last_tick_factor[0])
            factors_idx[rf_idx, 0] = idx_last
            idx_first = idx_last

        return factors_idx

    def __create_tpb_to_ticks_array(self, rest: bool = False) -> dict[int, np.ndarray]:
        r"""
        Create arrays of the times in ticks of the time tokens of the vocabulary.

        These time values following the ticks/beat value, which depends on the time
        signature.

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
            for tpb in self._tpb_per_ts.values()
        }

    def __create_tpb_tokens_to_ticks(
        self, rest: bool = False
    ) -> dict[int, dict[str, int]]:
        r"""
        Create the correspondences between times in tick and token value (str).

        These correspondences vary following the ticks/beat value, which depends on the
        time signature.

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
            for tpb in self._tpb_per_ts.values()
        }

    def __create_tpb_ticks_to_tokens(self) -> dict[int, dict[int, str]]:
        r"""
        Create the correspondences between times in tick and token value (str).

        These correspondences vary following the ticks/beat value, which depends on the
        time signature.

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
        r"""
        Convert a time token value of the form beat.position.resolution, in ticks.

        This method is used to decode time tokens such as *Duration*, *TimeShift* or
        *Rest*.

        :param token_duration: Duration / TimeShift token value.
        :param ticks_per_beat: number of ticks in a beat. This depends on the current
            time signature, and is equal to the Score's time division if the denominator
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
        r"""
        Convert a duration in ticks into a sequence of *TimeShift*/*Rest* values.

        This method is not used for *Duration* tokens, as their values are rounded to
        the closest values in. It is however used to create successions of time values
        for *TimeShift* and *Rest* tokens.

        :param duration: duration in tick to convert.
        :param ticks_per_beat: number of ticks in a beat. This depends on the current
            time signature, and is equal to the Score's time division if the denominator
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
        r"""
        Create the rests of the vocabulary as tuples of integers.

        The tuples have the form ``(beat, pos, res)`` where ``beat`` is the number
        of beats, ``pos`` the number of "positions" and ``res`` the beat resolution
        considered (positions per beat).
        It follows the same data representation than for duration and time shifts.

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
        r"""
        Create the tempos of the vocabulary as a float number array.

        The ``self.config.num_tempos`` tempos are distributed in the
        ``self.config.tempo_range`` using either log or linear scaled values based
        on the value of ``self.config.log_tempos``.

        :return: the tempos as a numpy array.
        """
        tempo_fn = np.geomspace if self.config.log_tempos else np.linspace
        return tempo_fn(*self.config.tempo_range, self.config.num_tempos).round(2)

    def __create_time_signatures(self) -> list[tuple]:
        r"""
        Create time signatures of the vocabulary, as tuples of integers.

        The tuples have the form ``(num_beats, beat_res)`` where ``num_beats`` is the
        number of beats per bar.
        Example: ``(3, 4)`` means one bar is 3 beat long and each beat is a quarter
        note.

        :return: the time signatures.
        """
        time_signature_range = self.config.time_signature_range

        time_signatures = []
        for beat_res, beats in time_signature_range.items():
            if beat_res <= 0 or not math.log2(beat_res).is_integer():
                msg = (
                    f"The beat resolution ({beat_res}) in time signature must be a"
                    f"power of 2."
                )
                raise ValueError(msg)

            time_signatures.extend([(num_beats, beat_res) for num_beats in beats])

        return time_signatures

    def __create_pitch_bends(self) -> np.ndarray:
        r"""
        Create the pitch bend values of the vocabulary as numpy array.

        :return: the pitch bend values.
        """
        return np.linspace(*self.config.pitch_bend_range, dtype=np.int32)

    @staticmethod
    def _parse_token_time_signature(token_time_sig: str) -> tuple[int, int]:
        r"""
        Convert a time signature token value of the form x/x into a tuple of integers.

        :param token_time_sig: TimeSig token value.
        :return: the numerator and denominator of a time signature.
        """
        numerator, denominator = map(int, token_time_sig.split("/"))
        return numerator, denominator

    def score_has_time_signatures_not_in_vocab(self, score: Score) -> bool:
        r"""
        Check if a ``symusic.Score`` contains unsupported time signatures.

        :param score: ``symusic.Score`` object.
        :return: boolean indicating whether the score can be processed by the tokenizer.
        """
        if self.config.use_time_signatures:
            for time_sig in score.time_signatures:
                if (
                    time_sig.numerator,
                    time_sig.denominator,
                ) not in self.time_signatures:
                    return True
        return False

    def learn_bpe(self, *args, **kwargs) -> Score:  # noqa: D102, ANN002
        warnings.warn(
            "miditok: The `learn_bpe` method had been renamed `train`. It is now "
            "depreciated and will be removed in future updates.",
            stacklevel=2,
        )
        return self.train(*args, **kwargs)

    def train(
        self,
        vocab_size: int,
        model: Literal["BPE", "Unigram", "WordPiece"] | _tok_models.Model | None = None,
        iterator: Iterable | None = None,
        files_paths: Sequence[Path] | None = None,
        **kwargs,
    ) -> None:
        r"""
        Train the tokenizer to build its vocabulary with BPE, Unigram or WordPiece.

        The data used for training can either be given through the ``iterator``
        argument as an iterable object yielding strings, or by ``files_paths`` as a
        list of paths to music files that will be tokenized.
        You can read the Hugging Face `🤗tokenizers documentation
        <https://huggingface.co/docs/tokenizers/index>`_, and `🤗tokenizers course
        <https://huggingface.co/course/chapter6/2?fw=pt>`_ for more details about the
        ``iterator`` and input type.

        If splitting the token sequences per bar or beat, a
        `"Metaspace" <https://huggingface.co/docs/tokenizers/api/pre-tokenizers#tokenizers.pre_tokenizers.Metaspace>`_
        pre-tokenizer and decoder will be used. Each chunk of tokens will be prepended
        with a special "▁" (U+2581) character to mark its beginning, as would be a word.

        **A few considerations to note:**

        1. The WordPiece model has a ``max_input_chars_per_word`` attribute, which
        controls the maximum number of "base tokens" a sequence of ids can contain until
        it discards and replaces it with a predefined "unknown" token (``unk_token``
        model attribute). This means that, depending on the base sequence lengths of
        your files, the tokenizer will likely discard them. This can be addressed by
        either: 1) splitting the token sequence per bars or beats before encoding ids
        (highly recommended) into smaller subsequences whose lengths will likely be
        lower to the model's ``max_input_chars_per_word`` attribute; 2) set the model's
        ``max_input_chars_per_word`` attribute to a value higher than most of the
        sequences of ids encoded by the WordPiece model.
        A high ``max_input_chars_per_word`` value will however drastically increase the
        encoding and decoding times, reducing its interest. The default values set by
        MidiTok are ``400`` when splitting ids in bar subsequences and ``100`` when
        splitting ids in beat subsequences.
        The ``max_input_chars_per_word`` and ``unk_token`` model attributes can be set
        by referencing them in the keyword arguments of this method (``kwargs``).
        2. The Hugging Face Unigram model training `is not 100% deterministic
        <https://github.com/huggingface/tokenizers/issues/668>`_. As such and if you are
        using Unigram, you should train your tokenizer only once before using it to save
        tokenized files or train a model. Otherwise, some token ids might be swapped,
        resulting in incoherent encodings-decodings.

        **The training progress bar will not appear with non-proper terminals.**
        (cf `GitHub issue <https://github.com/huggingface/tokenizers/issues/157>`_ )

        :param vocab_size: size of the vocabulary to learn / build.
        :param model: backbone model to use to train the tokenizer. MidiTok relies on
            the Hugging Face tokenizers library, and supports the ``BPE``, ``Unigram``
            and ``WordPiece`` models. This argument can be either a string indicating
            the model to use, an already initialized model, or ``None`` if you want to
            retrain a tokenizer already trained. (default: ``None``, default to
            ``BPE`` if the tokenizer is not already trained, keeps the same model
            otherwise)
        :param iterator: an iterable object yielding the training data, as lists of
            string. It can be a list or a Generator. This iterator will be passed to
            the model for training. It musts implement the ``__len__`` method. If
            None is given, you must use the ``tokens_paths`` argument. (default: None)
        :param files_paths: paths of the music files to load and use. (default: None)
        :param kwargs: any additional argument to pass to the trainer or model. See the
            `tokenizers docs <https://huggingface.co/docs/tokenizers/api/trainers>`_
            for more details.
        """
        # Checks the arguments/config are compatible for training
        if self.is_multi_voc:
            warnings.warn(
                "This tokenizer is based on multiple vocabularies/embedding pooling."
                "It therefore cannot be trained. Skipping `tokenizer.train` function"
                "call.",
                stacklevel=2,
            )
            return
        if iterator is None and files_paths is None:
            msg = (
                "You must give an iterator or a list of paths to tokens to train the"
                "tokenizer."
            )
            raise ValueError(msg)
        if vocab_size <= len(self._vocab_base):
            warnings.warn(
                f"miditok - tokenizer.train: `vocab_size` ({vocab_size}) need to be "
                f"higher than the number of base tokens ({len(self._vocab_base)}). "
                f"Skipping tokenizer training.",
                stacklevel=2,
            )
            return

        # If no iterator, loads tokens / samples to analyze
        if iterator is None:
            iterator = TokTrainingIterator(self, files_paths)

        # Define the initial alphabet
        initial_alphabet = {
            chr(i + CHR_ID_START): i
            for tok, i in self._vocab_base.items()
            if not tok.endswith(UNIGRAM_SPECIAL_TOKEN_SUFFIX)
        }  # if to discard special tokens with Unigram

        # Define the model
        # A `tokenizers.Tokenizer` can feature: a normalizer, pre-tokenizer, model,
        # post-processor and decoder. We (in MidiTok) are only interested in the
        # model part, as other components will only perform operations on bytes and are
        # only relevant for text. MidiTok already "normalize" (preprocess music file),
        # pre-tokenize (split on bars/beats), and decoding is done differently for each
        # music tokenization.
        # Keep current model if `arg` is None:
        retraining = False
        if self._model is not None and model is None:
            tokenizer = self.__reload_hf_tokenizer(self._model)
            retraining = True
        # User provided a HF model
        elif isinstance(model, _tok_models.Model):
            tokenizer = _HFTokenizer(model)
        # User provided a str class model
        elif isinstance(model, str) or model is None:
            if model is None:  # default
                model = DEFAULT_TRAINING_MODEL_NAME
            model_kwargs = {"vocab": [] if model == "Unigram" else initial_alphabet}
            if model in ("BPE", "WordPiece") and self.config.encode_ids_split != "no":
                model_kwargs["continuing_subword_prefix"] = ""
                model_kwargs["end_of_word_suffix"] = ""
            if model == "BPE":
                model_kwargs["merges"] = []
            elif model == "WordPiece":
                model_kwargs["unk_token"] = kwargs.pop(
                    "unk_token", chr(self.pad_token_id + CHR_ID_START)
                )
                model_kwargs["max_input_chars_per_word"] = kwargs.pop(
                    "max_input_chars_per_word",
                    WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BEAT
                    if self.config.encode_ids_split == "beat"
                    else WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BAR,
                )
            tokenizer = _HFTokenizer(getattr(_tok_models, model)(**model_kwargs))
            if self.config.encode_ids_split != "no":
                # TODO the special char ▁ (U+2581) shouldn't be used elsewhere in vocab
                # chr(9601) = ▁
                tokenizer.pre_tokenizer = _pre_tokenizers.Metaspace()
                tokenizer.decoder = _decoders.Metaspace()
        else:
            msg = (
                "miditok - tokenizer.train: the `model` argument must be a str specify "
                "the model to use ('BPE', 'Unigram', 'WordPiece'), an already"
                "initialized model or a `None` to either resume training or default to "
                f"{DEFAULT_TRAINING_MODEL_NAME}."
            )
            raise ValueError(msg)

        # Converts model to json
        tokenizer_json = json.loads(tokenizer.to_str())
        # Remove added tokens for now (uses IDs of tokens)
        added_tokens = tokenizer_json.pop("added_tokens")
        model_name = tokenizer_json["model"]["type"]

        # Warnings
        if retraining and model_name == "Unigram":
            warnings.warn(
                "miditok - tokenizer.train: You are retraining a tokenizer with "
                "Unigram. The Hugging Face Unigram model training is not 100% "
                "deterministic. As such and if you are using it, you should train your "
                "tokenizer only once before using it to save tokenized files or train "
                "a model. Otherwise some token ids might be swapped, resulting in "
                "incoherent encodings-decodings.",
                stacklevel=2,
            )
        if (
            model == "WordPiece" or isinstance(model, _tok_models.WordPiece)
        ) and self.config.encode_ids_split == "no":
            warnings.warn(
                "miditok - tokenizer.train: you are training a WordPiece tokenizer "
                "without splitting the token ids per bars or beats. It is recommended "
                "do so, as the tokenizer will either 1) replace ids sequences longer "
                "than its `max_input_chars_per_word` attribute ("
                f"{tokenizer_json['model']['max_input_chars_per_word']}) with its "
                f"`unk_token` attribute {tokenizer_json['model']['unk_token']}) thus "
                "not guaranteeing keeping data integrity when encoding token ids, "
                "unless you 2) set a large enough `max_input_chars_per_word` attribute "
                "being greater than the maximum base tokens sequence length of your "
                "data, which is likely to hurt performances and result in larger"
                "encoding-decoding time.",
                stacklevel=2,
            )

        # Get the special tokens from the current tokenizer if none are specified.
        special_tokens, special_tokens_str = [], []
        for added_token in added_tokens:
            special = added_token.pop("special", None)
            _ = added_token.pop("id", None)
            if not special:
                continue
            special_tokens.append(AddedToken(**added_token))
            special_tokens_str.append(added_token["content"])
        # Make sure all the special tokens of the tokenizer are referenced
        for token_id, token_str in zip(self.special_tokens_ids, self.special_tokens):
            # For Unigram, we have to make an exception for special tokens. A special
            # token cannot be just a character from the initial vocabulary. As such, we
            # prepend and append a special character to the byte of each special token.
            # Example: Pad_None = "!" becomes "!!!", BOS_None = "#" becomes "!#!"
            # As a result, each special token will take two "slots" in the vocabulary:
            # one for its distinct byte and one for its byte combination.
            special_token_byte = self._vocab_base_id_to_byte[token_id]
            if model_name == "Unigram" and not token_str.endswith(
                UNIGRAM_SPECIAL_TOKEN_SUFFIX
            ):
                special_char = chr(CHR_ID_START)
                special_token_byte = f"{special_char}{special_token_byte}{special_char}"
            if special_token_byte not in special_tokens_str:
                if model_name == "Unigram":
                    special_token_str = (
                        f"{self.__vocab_base_inv[token_id]}"
                        f"{UNIGRAM_SPECIAL_TOKEN_SUFFIX}"
                    )
                    if special_token_str not in self.vocab:
                        self.add_to_vocab(
                            special_token_str,
                            special_token=True,
                            byte_=special_token_byte,
                        )
                special_tokens.append(AddedToken(special_token_byte))

        # Trainer needs to know the end of word / continuing subword thingies in BPE
        if model_name in ["BPE", "WordPiece"]:
            args = ["continuing_subword_prefix"]
            if model_name == "BPE":
                args.append("end_of_word_suffix")
            for arg in args:
                if arg not in kwargs and tokenizer_json["model"][arg] is not None:
                    kwargs[arg] = tokenizer_json["model"][arg]
        elif model_name == "Unigram" and tokenizer_json["model"]["unk_id"] is not None:
            if "max_piece_length" not in kwargs:
                kwargs["max_piece_length"] = (
                    UNIGRAM_MAX_INPUT_CHARS_PER_WORD_BEAT
                    if self.config.encode_ids_split == "beat"
                    else UNIGRAM_MAX_INPUT_CHARS_PER_WORD_BAR
                )
            unk_id = tokenizer_json["model"]["unk_id"]
            kwargs["unk_token"] = tokenizer_json["model"]["vocab"][unk_id][0]

        # Trains the tokenizer
        name_trainer = f"{'Bpe' if model_name == 'BPE' else model_name}Trainer"
        trainer = getattr(_tok_trainers, name_trainer)(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=special_tokens,
            initial_alphabet=list(initial_alphabet.keys()),
            **kwargs,
        )
        tokenizer.train_from_iterator(iterator, length=len(iterator), trainer=trainer)
        self._model = tokenizer

        # Update _vocab_learned_bytes_to_tokens for faster decoding
        self._vocab_learned_bytes_to_tokens = {}
        self.__create_vocab_learned_bytes_to_tokens()

    @staticmethod
    def __reload_hf_tokenizer(tokenizer: _HFTokenizer) -> _HFTokenizer:
        # Converts model to json
        tokenizer_json = json.loads(tokenizer.to_str())

        # Remove vocab
        # Apr 16th 2024 - MidiTok v3.0.3: when resuming training, we restart all over
        # from a clean vocabulary. As tokenizers v0.19.0, resuming a training does
        # not further reduce/compress the sequences even with a larger vocabulary.
        if tokenizer_json["model"]["type"] == "BPE":
            tokenizer_json["model"]["vocab"] = {}
            tokenizer_json["model"]["merges"] = []
        elif tokenizer_json["model"]["type"] == "Unigram":
            tokenizer_json["model"]["vocab"] = []
            if tokenizer_json["model"]["unk_id"] is not None:
                unk_id = tokenizer_json["model"]["unk_id"]
                unk_token = tokenizer_json["model"]["vocab"][unk_id][0]
                tokenizer_json["model"]["unk_id"] = 0
                tokenizer_json["model"]["vocab"] = [[unk_token, 0.0]]
        elif tokenizer_json["model"]["type"] == "WordPiece":
            tokenizer_json["model"]["vocab"] = {}
        else:
            msg = (
                "This method does not support this type of tokenizer (found "
                f"{tokenizer_json['model']['type']}) only BPE, Unigram or WordPiece."
            )
            raise ValueError(msg)

        return _HFTokenizer.from_str(json.dumps(tokenizer_json))

    def __create_vocab_learned_bytes_to_tokens(self) -> None:
        try:
            continuing_subword_prefix = self._model.model.continuing_subword_prefix
        except AttributeError:
            continuing_subword_prefix = ""
        if continuing_subword_prefix is None:
            continuing_subword_prefix = ""
        try:
            end_of_word_suffix = self._model.model.end_of_word_suffix
        except AttributeError:
            end_of_word_suffix = ""
        if end_of_word_suffix is None:
            end_of_word_suffix = ""

        for k in self._model.get_vocab():
            key_ = k
            if continuing_subword_prefix != "" and key_.startswith(
                continuing_subword_prefix
            ):
                key_ = key_[len(continuing_subword_prefix) :]
            if end_of_word_suffix != "" and key_.endswith(end_of_word_suffix):
                key_ = key_[: -len(end_of_word_suffix)]
            if isinstance(self._model.pre_tokenizer, _pre_tokenizers.Metaspace):
                replacement = self._model.pre_tokenizer.replacement
                if key_.startswith(replacement):
                    key_ = key_[len(replacement) :]
            self._vocab_learned_bytes_to_tokens[k] = [
                self._vocab_base_byte_to_token[b] for b in key_
            ]

    def apply_bpe(self, *args, **kwargs) -> Score:  # noqa: D102, ANN002
        warnings.warn(
            "miditok: The `apply_bpe` method had been renamed `encode_token_ids`. It is"
            " now depreciated and will be removed in future updates.",
            stacklevel=2,
        )
        return self.encode_token_ids(*args, **kwargs)

    def encode_token_ids(self, seq: TokSequence | list[TokSequence]) -> None:
        """
        Encode a :class:`miditok.TokSequence` with BPE, Unigram or WordPiece.

        The method works inplace and only alters the sequence's ``.ids``.
        The method also works with lists of :class:`miditok.TokSequence`.
        If a list is given, the model will encode all sequences in one batch to speed up
        the operation.

        :param seq: :class:`miditok.TokSequence` to encode ids.
        """

        def _split_seq_bytes(seq__: TokSequence) -> list[str]:
            self.complete_sequence(seq__, complete_bytes=True)
            if self.config.encode_ids_split == "bar":
                return [subseq.bytes for subseq in seq__.split_per_bars()]
            if self.config.encode_ids_split == "beat":
                return [subseq.bytes for subseq in seq__.split_per_beats()]
            return [seq__.bytes]

        # No recursivity as we can leverage batching here
        if isinstance(seq, list):
            all_bytes = [_split_seq_bytes(seq_) for seq_ in seq]
            encoded_tokens = self._model.encode_batch(all_bytes, is_pretokenized=True)
            for seq_, ids_encoded in zip(seq, encoded_tokens):
                seq_.ids = ids_encoded.ids
                seq_.are_ids_encoded = True
        else:
            all_bytes = _split_seq_bytes(seq)
            encoded_tokens = self._model.encode(all_bytes, is_pretokenized=True)
            seq.ids = encoded_tokens.ids
            seq.are_ids_encoded = True

    def decode_bpe(self, *args, **kwargs) -> Score:  # noqa: D102, ANN002
        warnings.warn(
            "miditok: The `decode_bpe` method had been renamed `decode_token_ids`. It "
            "is now depreciated and will be removed in future updates.",
            stacklevel=2,
        )
        return self.decode_token_ids(*args, **kwargs)

    def decode_token_ids(self, seq: TokSequence | list[TokSequence]) -> None:
        r"""
        Decode the ids of a :class:`miditok.TokSequence` with BPE, Unigram or WordPiece.

        This method only modifies the ``.ids`` attribute of the input sequence(s)
        and does not complete it. This method can be used recursively on lists of
        :class:`miditok.TokSequence`.

        :param seq: token sequence to decompose.
        """
        # This method directly convert encoded ids to base tokens
        if isinstance(seq, list):
            [self.decode_token_ids(seq_) for seq_ in seq]

        elif isinstance(seq, TokSequence) and seq.are_ids_encoded:
            encoded_bytes = [self._model.id_to_token(id_) for id_ in seq.ids]
            decoded_tokens = [
                self._vocab_learned_bytes_to_tokens[byte_] for byte_ in encoded_bytes
            ]
            decoded_tokens = [
                item for sublist in decoded_tokens for item in sublist
            ]  # flatten
            seq.tokens = decoded_tokens
            seq.ids = self._tokens_to_ids(decoded_tokens)
            seq.are_ids_encoded = False

    def tokenize_midi_dataset(self, *args, **kwargs) -> Score | list[Score]:  # noqa: D102, ANN002
        warnings.warn(
            "miditok: The `tokenize_midi_dataset` method had been renamed "
            "`tokenize_dataset`. It is now depreciated and will be removed in future "
            "updates.",
            stacklevel=2,
        )
        return self.tokenize_dataset(*args, **kwargs)

    def tokenize_dataset(
        self,
        files_paths: str | Path | Sequence[str | Path],
        out_dir: str | Path,
        overwrite_mode: bool = True,
        validation_fn: Callable[[Score], bool] | None = None,
        save_programs: bool | None = None,
        verbose: bool = True,
    ) -> None:
        r"""
        Tokenize a dataset or list of music files and save them in Json files.

        The resulting json files will have an ``ids`` entry containing the token ids.
        The format of the ids will correspond to the format of the tokenizer
        (``tokenizer.io_format``). Note that the file tree of the source files, up to
        the deepest common root directory if ``files_paths`` is given as a list of
        paths, will be reproducing in ``out_dir``. The config of the tokenizer will be
        saved as a file named ``tokenizer_config_file_name`` (default:
        ``tokenizer.json``) in the ``out_dir`` directory.

        :param files_paths: paths of the music files (MIDI, abc). It can also be a path
            to a directory, in which case this method will recursively find the MIDI and
            abc files within (.mid, .midi and .abc extensions, case insensitive).
        :param out_dir: output directory to save the converted files.
        :param overwrite_mode: if True, will overwrite files if they already exist when
            trying to save the new ones created by the method. This is enabled by
            default, as it is good practice to use dedicated directories for each
            tokenized dataset. If False, if a file already exist, the new one will be
            saved in the same directory, with the same name with a number appended at
            the end. Both token files and tokenizer config are concerned.
            (default: ``True``)
        :param validation_fn: a function checking if a music file is valid validates
            your conditions (e.g. time signature, minimum/maximum length,
            instruments...). (default: ``None``)
        :param save_programs: will save the programs of the tracks of the files as an
            entry in the Json file. This option is probably unnecessary when using a
            multitrack tokenizer (`config.use_programs`), as the program information is
            present within the tokens, and that the tracks having the same programs are
            likely to have been merged. (default: ``False`` if ``config.use_programs``,
            else ``True``)
        :param verbose: will throw warnings of errors when loading files, or if
            some files content is incorrect or need your attention. (default: ``True``)
        """
        self._verbose = verbose
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # User gave a path to a directory, we'll scan it to find MIDI/abc files
        if not isinstance(files_paths, Sequence):
            if isinstance(files_paths, str):
                files_paths = Path(files_paths)
            root_dir = files_paths
            files_paths = [
                path
                for path in files_paths.glob("**/*")
                if path.suffix in SUPPORTED_MUSIC_FILE_EXTENSIONS
            ]
        # User gave a list of paths, we need to find the root / deepest common subdir
        else:
            root_dir = get_deepest_common_subdir(files_paths)

        if save_programs is None:
            save_programs = not self.config.use_programs

        # Tokenizing
        # Note: tests with multiprocessing show significant slower runtime with 4
        # workers.
        desc = f"Tokenizing music files ({'/'.join(list(out_dir.parts[-2:]))})"
        for file_path in tqdm(files_paths, desc=desc):
            # Some files can contain errors, if so the loop continues
            file_path = Path(file_path)
            try:
                score = Score(file_path)
            except FileNotFoundError:
                if self._verbose:
                    warnings.warn(f"File not found: {file_path}", stacklevel=2)
                continue
            except SCORE_LOADING_EXCEPTION:
                continue

            # Passing the Score to validation tests if given
            if validation_fn is not None and not validation_fn(score):
                continue

            # Tokenizing the Score
            tokens = self.encode(score)

            # Set output file path
            out_path = out_dir / file_path.resolve().parent.relative_to(root_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            out_path /= f"{file_path.stem}.json"

            # If non-overwrite, set the new file name
            if not overwrite_mode and out_path.is_file():
                i = 1
                while out_path.is_file():
                    out_path = out_path.parent / f"{file_path.stem}_{i}.json"
                    i += 1

            # Save the tokens as JSON
            self.save_tokens(
                tokens,
                out_path,
                get_score_programs(score) if save_programs else None,
            )

        # Set it back to False
        self._verbose = False

    def tokens_errors(
        self,
        tokens: TokSequence | list[TokSequence] | list[int | list[int]] | np.ndarray,
    ) -> float | list[float]:
        r"""
        Return the ratio of errors of prediction in a sequence of tokens.

        Check if a sequence of tokens is made of good token types successions and
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
        if len(tokens) == 0:
            return 0

        num_tok_predicted = len(tokens)  # used to norm the score
        if self.is_trained:
            self.decode_token_ids(tokens)
        self.complete_sequence(tokens)

        # Compute number of errors and norm by number of tokens predicted
        return self._tokens_errors(tokens.tokens) / num_tok_predicted

    def _tokens_errors(self, tokens: list[str | list[str]]) -> int:
        r"""
        Return the number of errors in a sequence of tokens.

        The method checks if a sequence of tokens is made of good token types
        successions and values. The number of errors should not be higher than the
        number of tokens.

        This method is intended to be overridden by tokenizer classes. The
        implementation in the ``MusicTokenizer`` class will check token types,
        duplicated notes and time errors. It works for ``REMI``, ``TSD`` and
        ``Structured``.

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
        previous_pitch_onset = dict.fromkeys(self.config.programs, -128)
        previous_pitch_chord = dict.fromkeys(self.config.programs, -128)
        note_tokens_types = ["Pitch", "NoteOn", "PitchDrum"]
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
                if token == "Bar_None":  # reset
                    current_pos = -1
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in ["TimeShift", "Time-Shift", "Rest"]:
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in note_tokens_types:
                    if event_type in {"Pitch", "NoteOn", "PitchDrum"}:
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
        r"""
        Save tokens as a JSON file.

        In order to reduce disk space usage, **only the ids are saved**. Use ``kwargs``
        to save any additional information within the JSON file.

        :param tokens: tokens, as list, numpy array, torch or tensorflow Tensor.
        :param path: path of the file to save.
        :param programs: (optional), programs of the associated tokens, should be given
            as a tuples (int, bool) for (program, is_drum).
        :param kwargs: any additional information to save within the JSON file.
        """
        ids = []
        ids_encoded = None

        if isinstance(tokens, TokSequence):
            if len(tokens.ids) == 0:
                self.complete_sequence(tokens)
            ids_encoded = tokens.are_ids_encoded
            ids = tokens.ids
        elif isinstance(tokens, list) and len(tokens) == 0:
            pass
        elif isinstance(tokens[0], TokSequence):
            ids_encoded = []
            for seq in tokens:
                if len(seq.ids) == 0:
                    self.complete_sequence(seq)
                ids_encoded.append(seq.are_ids_encoded)
                ids.append(seq.ids)
        else:
            ids = convert_ids_tensors_to_list(tokens)

        if "ids_encoded" not in kwargs and ids_encoded is not None:
            kwargs["ids_encoded"] = ids_encoded

        with Path(path).open("w") as outfile:
            dic = {"ids": ids, **kwargs}
            if programs is not None:
                dic["programs"] = programs
            json.dump(dic, outfile)

    def load_tokens(
        self, path: str | Path, raw: bool = False
    ) -> TokSequence | list[TokSequence] | dict:
        r"""
        Load tokens saved as JSON files.

        :param path: path of the file to load.
        :param raw: if given ``True``, will return the raw content of the json file.
            (default: ``False``)
        :return: the tokens, with the associated information saved with.
        """
        with Path(path).open() as file:
            json_content = json.load(file)

        if raw:
            return json_content

        return self._convert_sequence_to_tokseq(json_content["ids"])

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        Save the tokenizer in local a directory.

        Overridden from ``huggingface_hub.ModelHubMixin``.
        Since v0.21 this method will automatically save ``self.config`` on after
        calling ``self._save_pretrained``, which is unnecessary in our case.

        :param save_directory: Path to directory in which the model weights and
            configuration will be saved.
        :param push_to_hub: Whether to push your model to the Huggingface Hub after
            saving it.
        :param repo_id: ID of your repository on the Hub. Used only if
            `push_to_hub=True`. Will default to the folder name if not provided.
        :param push_to_hub_kwargs: Additional key word arguments passed along to the
            [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files (framework-specific)
        self._save_pretrained(save_directory)

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def _save_pretrained(self, *args, **kwargs) -> None:  # noqa: ANN002
        # called by `ModelHubMixin.from_pretrained`.
        self.save(*args, **kwargs)

    def save_params(self, *args, **kwargs) -> None:  # noqa: ANN002
        """
        **DEPRECIATED:** save a tokenizer as a JSON file (calling ``tokenizer.save``).

        :param args: positional arguments.
        :param kwargs: keyword arguments.
        """
        warnings.warn(
            "miditok: The `save_params` method had been renamed `save`. It is now "
            "depreciated and will be removed in future updates.",
            stacklevel=2,
        )
        return self.save(*args, **kwargs)

    def to_dict(self) -> dict:
        """Return the serializable dictionary form of the tokenizer."""
        params = {
            "config": self.config.to_dict(serialize=True),
            "tokenization": self.__class__.__name__,
            "miditok_version": CURRENT_MIDITOK_VERSION,
            "symusic_version": CURRENT_SYMUSIC_VERSION,
            "hf_tokenizers_version": CURRENT_TOKENIZERS_VERSION,
        }
        if self.is_trained:  # saves whole vocab if trained
            params["_vocab_base"] = self._vocab_base
            params["_model"] = self._model.to_str()
            params["_vocab_base_byte_to_token"] = self._vocab_base_byte_to_token
        return params

    def save(
        self,
        out_path: str | Path,
        additional_attributes: dict | None = None,
        filename: str | None = DEFAULT_TOKENIZER_FILE_NAME,
    ) -> None:
        r"""
        Save tokenizer in a Json file.

        This can be useful to keep track of how a dataset has been tokenized.

        :param out_path: output path to save the file. This can be either a path to a
            file (with a name and extension), or a path to a directory in which case
            the ``filename`` argument will be used.
        :param additional_attributes: any additional information to store in the config
            file. It can be used to override the default attributes saved in the parent
            method. (default: ``None``)
        :param filename: name of the file to save, to be used in case ``out_path`` leads
            to a directory. (default: ``"tokenizer.json"``)
        """
        tokenizer_dict = self.to_dict()

        if additional_attributes:
            tokenizer_dict.update(additional_attributes)

        out_path = Path(out_path)
        if out_path.is_dir() or "." not in out_path.name:
            out_path /= filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as outfile:
            json.dump(tokenizer_dict, outfile, indent=4)

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
    ) -> MusicTokenizer:
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

        # Checking config file tokenization
        with Path(params_path).open() as file:
            tokenization = json.load(file)["tokenization"]
        cls_name = cls.__name__
        if cls_name not in ["MusicTokenizer", tokenization]:
            warnings.warn(
                ".from_pretrained called with an invalid class name. The current class"
                f"is {cls_name} whereas the config file comes from a {tokenization} "
                f"tokenizer. Returning an instance of {tokenization}.",
                stacklevel=2,
            )

        if cls_name == tokenization:
            return cls(params=params_path)

        miditok_module = sys.modules[".".join(__name__.split(".")[:-1])]
        return getattr(miditok_module, tokenization)(params=params_path)

    def _load_from_json(self, file_path: str | Path) -> None:
        r"""
        Load the parameters of the tokenizer from a config file.

        This method is not intended to be called outside __init__, when creating a
        tokenizer.

        :param file_path: path to the tokenizer JSON file.
        """
        with Path(file_path).open() as param_file:
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
            if key == "_vocab_base":
                self._vocab_base = value
                self.__vocab_base_inv = {v: k for k, v in value.items()}
                continue
            if key == "_model":
                # using 🤗tokenizers builtin method
                self._model = _HFTokenizer.from_str(value)
                continue
            if key == "_vocab_base_byte_to_token":
                self._vocab_base_byte_to_token = value
                token_to_byte = {v: k for k, v in value.items()}
                self._vocab_base_id_to_byte = {
                    i: token_to_byte[tok] for tok, i in self._vocab_base.items()
                }
                self.__create_vocab_learned_bytes_to_tokens()
                continue
            if key == "config":
                if "chord_maps" in value:
                    value["chord_maps"] = {
                        chord_quality: tuple(chord_map)
                        for chord_quality, chord_map in value["chord_maps"].items()
                    }
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
            if key in config_attributes:
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
            if key == "has_bpe":
                # For config files < v3.0.3 before the attribute becomes a property
                continue

            setattr(self, key, value)

    @property
    def is_multi_voc(self) -> bool:
        """
        Indicate if the tokenizer uses embedding pooling / have multiple vocabularies.

        :return: ``True`` is the tokenizer uses embedding pooling else ``False``.
        """
        return isinstance(self._vocab_base, list)

    @property
    def io_format(self) -> tuple[str, ...]:
        """
        Return the i/o format of the tokenizer.

        The characters for each dimension returned are:
        * ``I``: track or instrument;
        * ``T``: token, or time step;
        * ``C``: class of token, when using embedding pooling.

        :return: i/o format of the tokenizer, as a tuple of strings which represent:
        """
        format_ = []
        if not self.one_token_stream:
            format_.append("I")
        format_.append("T")
        if self.is_multi_voc:
            format_.append("C")

        return tuple(d for d in format_)

    def __call__(
        self,
        obj: Score | TokSequence | list[TokSequence, int, list[int]] | np.ndarray,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> TokSequence | list[TokSequence] | Score:
        r"""
        Tokenize a music file (MIDI/abc), or decode tokens into a ``symusic.Score``.

        Calling a tokenizer allows to directly convert a music file (MIDI/abc) to tokens
        or vice-versa. The method automatically detects ``symusic.Score`` and
        :class:`miditok.TokSequence` objects, as well as paths to music or json files.
        It will call the :py:func:`miditok.MusicTokenizer.encode` if you provide a
        ``symusic.Score`` object or path to a music file, or the
        :py:func:`miditok.MusicTokenizer.decode` method otherwise.

        :param obj: a `symusic.Score` object, a :class:`miditok.TokSequence` object, or
            a path to a music or tokens json file.
        :return: the converted object.
        """
        # Tokenize `Score`
        if isinstance(obj, Score):
            return self.encode(obj, *args, **kwargs)

        # Path provided: Encode/decode a file
        if isinstance(obj, (str, Path)):
            obj = Path(obj)
            # tokens
            if obj.suffix == "json":
                tokens = self.load_tokens(obj)
                return self.decode(tokens["ids"], *args, **kwargs)
            # music file
            return self.encode(obj, *args, **kwargs)

        # Depreciated miditoolkit object
        if MidiFile is not None and isinstance(obj, MidiFile):
            warnings.warn(
                "You are using a depreciated `miditoolkit.MidiFile` object. MidiTok"
                "is now (>v3.0.0) using symusic.Score as MIDI backend. Your file will"
                "be converted on the fly, however please consider using symusic.",
                stacklevel=2,
            )
            return self.encode(miditoolkit_to_symusic(obj), *args, **kwargs)

        # Decode tokens, may be a TokSequence, numpy array or tensor
        return self.decode(obj, *args, **kwargs)

    def __len__(self) -> int:
        r"""
        Return the length of the vocabulary.

        If the tokenizer uses embedding pooling/have multiple vocabularies, it will
        return the **sum** of their lengths. If the tokenizer has been trained, this
        method returns the length of its model's vocabulary, i.e. the proper number of
        possible token ids. Otherwise, it will return the length of the base
        vocabulary. Use the :py:func:`miditok.MusicTokenizer.len` property
        (``tokenizer.len``) to get the list of lengths.

        :return: length of the vocabulary.
        """
        if self.is_multi_voc:
            return sum([len(v) for v in self.vocab])
        if self.is_trained:
            return len(self._model.get_vocab())
        return len(self.vocab)

    @property
    def len(self) -> int | list[int]:
        r"""
        Return the length of the vocabulary.

        If the tokenizer uses embedding pooling/have multiple vocabularies, it will
        return the **list** of their lengths. Use the
        :py:func:`miditok.MusicTokenizer.__len__` magic method
        (``len(tokenizer)``) to get the sum of the lengths.

        :return: length of the vocabulary.
        """
        return [len(v) for v in self.vocab] if self.is_multi_voc else len(self)

    def __repr__(self) -> str:
        """
        Return the representation of the tokenizer, indicating its vocab size and i/o.

        :return: representation of the tokenizer.
        """
        out_str = f"{self.len} tokens with {self.io_format} io format"

        # one_token_stream / multi-voc
        tmp = []
        if self.one_token_stream:
            tmp.append("one token stream")
        if self.is_multi_voc:
            tmp.append("multi-voc")
        if len(tmp) > 0:
            out_str += f" ({', '.join(tmp)})"

        # Trained
        if self.is_trained:
            out_str += f", trained with {self._model_name}"
        else:
            out_str += ", not trained"
        return out_str

    def __getitem__(
        self, item: int | str | tuple[int, int | str]
    ) -> str | int | list[int]:
        r"""
        Convert a token (int) to an event (str), or vice-versa.

        :param item: a token (int) or an event (str). For tokenizers with
            embedding pooling/multiple vocabularies ( `tokenizer.is_multi_voc` ), you
            must either provide a string (token) that is within all vocabularies (e.g.
            special tokens), or a tuple where the first element in the index of the
            vocabulary and the second the element to index.
        :return: the converted object.
        """
        if isinstance(item, tuple) and self.is_multi_voc:
            return self.__get_from_voc(item[1], item[0])
        if self.is_multi_voc and isinstance(item, str):
            if all(item in voc for voc in self.vocab):
                # It should be the same id for all sub-vocabs
                return self.vocab[0][item]

            msg = (
                "This tokenizer uses multiple vocabularies / embedding pooling. To"
                "index it you must either provide a token (string) that is within"
                "all vocabularies (e.g. special tokens), or a tuple where the"
                "first element in the index of the vocabulary and the second the"
                "element to index."
            )
            raise ValueError(msg)

        return self.__get_from_voc(item)

    def __get_from_voc(self, item: int | str, vocab_id: int | None = None) -> int | str:
        r"""
        Get an element from the vocabulary.

        The method handles both id (int) <--> token (str) ways.

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

    def __eq__(self, other: MusicTokenizer) -> bool:
        r"""
        Check that two tokenizers are identical.

        This is done by comparing their vocabularies, and configuration.

        :param other: tokenizer to compare.
        :return: True if the vocabulary(ies) are identical, False otherwise.
        """
        if not isinstance(other, type(self)):
            return False
        vocab_trained_eq = self.is_trained == other.is_trained
        if self.is_trained and other.is_trained:
            vocab_trained_eq = self._model.get_vocab() == other._model.get_vocab()
        return (
            self._vocab_base == other._vocab_base
            and vocab_trained_eq
            and self._vocab_base_byte_to_token == other._vocab_base_byte_to_token
            and self.config == other.config
        )
