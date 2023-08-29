"""
MIDI encoding base class and methods
"""
from abc import ABC, abstractmethod
import math
from pathlib import Path
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Union, Callable, Iterable, Optional, Any, Sequence

import numpy as np
from tqdm import tqdm
from miditoolkit import (
    MidiFile,
    Instrument,
    Note,
    TempoChange,
    TimeSignature,
    Pedal,
    PitchBend,
)
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .classes import Event, TokSequence, TokenizerConfig
from .utils import (
    remove_duplicated_notes,
    get_midi_programs,
    convert_ids_tensors_to_list,
    merge_same_program_tracks,
    detect_chords,
)
from .data_augmentation import data_augmentation_dataset
from .constants import (
    TIME_DIVISION,
    CURRENT_VERSION_PACKAGE,
    TIME_SIGNATURE,
    CHR_ID_START,
    PITCH_CLASSES,
    UNKNOWN_CHORD_PREFIX,
    MIDI_FILES_EXTENSIONS,
)


def convert_sequence_to_tokseq(
    tokenizer, input_seq, complete_seq: bool = True, decode_bpe: bool = True
) -> Union[TokSequence, List[TokSequence]]:
    r"""Converts a sequence into a :class:`miditok.TokSequence` or list of :class:`miditok.TokSequence`
    objects with the appropriate format of the tokenizer being used.

    :param tokenizer: tokenizer being used with the sequence.
    :param input_seq: sequence to convert. It can be a list of ids (integers), tokens (string) or events (Event).
        It can also be a Pytorch or TensorFlow tensor, or Numpy array representing ids.
    :param complete_seq: will complete the output sequence(s). (default: True)
    :param decode_bpe: if the input sequence contains ids, and that they contain BPE tokens, these tokens will
        be decoded. (default: True)
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
    if isinstance(arg[1][0], list):
        nb_seq_dims += 1
        if isinstance(arg[1][0][0], list):
            nb_seq_dims += 1

    # Check the number of dimensions is good
    # In case of no one_token_stream and one dimension short --> unsqueeze
    if not tokenizer.one_token_stream and nb_seq_dims == nb_io_dims - 1:
        print(
            f"The input sequence has one dimension less than expected ({nb_seq_dims} instead of "
            f"{nb_io_dims}). It is being unsqueezed to conform with the tokenizer's i/o format "
            f"({tokenizer.io_format})"
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
            if not tokenizer.is_multi_voc:
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
    r"""Decorator creating if necessary and completing a :class:`miditok.TokSequence` object before that the function
    is called. This decorator is made to be used by the :py:meth:`miditok.MIDITokenizer.tokens_to_midi` method.

    :param complete: will complete the sequence, i.e. complete its ``ids`` , ``tokens`` and ``events`` .
    :param decode_bpe: will decode BPE, if applicable. This step is performed before completing the sequence.
    """

    def decorator(function: Callable = None):
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


class MIDITokenizer(ABC):
    r"""MIDI tokenizer base class, containing common methods and attributes for all tokenizers.

    :param tokenizer_config: the tokenizer's configuration, as a :class:`miditok.classes.TokenizerConfig` object.
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: Union[str, Path] = None,
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
        # Determines how the tokenizer will handle multiple tracks: either each track as a single indepandant token
        # stream (False), or all the tracks as a single token stream (True).
        self.one_token_stream = False

        # Loading params, or initializing them from args
        if params is not None:
            # Will overwrite self.config
            self._load_params(params)
        else:
            # If no TokenizerConfig is given, we falls back to the default parameters
            if self.config is None:
                self.config = TokenizerConfig()
            assert (
                self.config.pitch_range[0] >= 0 and self.config.pitch_range[1] <= 128
            ), "You must specify a pitch_range between 0 and 127 (included, i.e. range.stop at 128)"
            assert (
                0 < self.config.nb_velocities < 128
            ), "You must specify a nb_velocities between 1 and 127 (included)"

        # Tweak the tokenizer's configuration and / or attributes before creating the vocabulary
        # This method is intended to be overridden by inheriting tokenizer classes
        self._tweak_config_before_creating_voc()

        # Set one_token_stream mode according to the config params
        if self.config.use_programs:
            self.one_token_stream = self.config.one_token_stream_for_programs

        # Init duration and velocity values
        self.durations = self.__create_durations_tuples()
        # [1:] so that there is no velocity_0
        self.velocities = np.linspace(
            0, 127, self.config.nb_velocities + 1, dtype=np.intc
        )[1:]
        self._first_beat_res = list(self.config.beat_res.values())[0]
        for beat_range, res in self.config.beat_res.items():
            if 0 in beat_range:
                self._first_beat_res = res
                break

        # Tempos
        self.tempos = np.zeros(1)
        if self.config.use_tempos:
            self.tempos = self.__create_tempos()

        # Rests
        self.rests = []
        if self.config.use_rests:
            max_rest_res = max(self.config.beat_res_rest.values())
            max_global_res = max(self.config.beat_res.values())
            assert max_rest_res <= max_global_res, (
                "The maximum resolution of the rests must be inferior or equal to the maximum resolution of the"
                f"global beat resolution (config.beat_res). Expected <= {max_global_res}, found {max_rest_res}"
            )
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

        # Keep in memory durations in ticks for seen time divisions so these values
        # are not calculated each time a MIDI is processed
        self._durations_ticks = {}
        self._rests_ticks = {}

        # Holds the tempo changes, time signature, time division and key signature of a
        # MIDI (being parsed) so that methods processing tracks can access them
        self._current_midi_metadata = {}  # needs to be updated each time a MIDI is read

    def _tweak_config_before_creating_voc(self):
        # called after setting the tokenizer's TokenizerConfig (.config). To be customized by tokenizer classes.
        pass

    @property
    def vocab(
        self,
    ) -> Union[Dict[str, int], List[Dict[str, int]]]:  # token (str) to its id (int)
        """Get the base vocabulary, as a dictionary linking tokens (str) to their ids (int).
        The different (hidden / protected) vocabulary attributes of the class are:

        * ``._vocab_base`` : Dict[str: int] token -> id - Registers all known base tokens.
        * ``.__vocab_base_inv`` : Dict[int: str] id -> token - Inverse of ``._base_vocab`` , to go the other way.
        * ``._vocab_base_id_to_byte`` : Dict[int: str] id -> byte - Link ids to their associated unique bytes.
        * ``._vocab_base_byte_to_token`` : Dict[str: str] - similar as above but for tokens.
        * ``._vocab_bpe_bytes_to_tokens`` : Dict[str: List[str]] byte(s) -> token(s) used to decode BPE.
        * ``._bpe_model.get_vocab()`` : Dict[str: int] byte -> id - bpe model vocabulary, based on unique bytes

        Before training the tokenizer with BPE, bytes are obtained by running ``chr(id)`` . After training,
        if we did start from an empty vocabulary, some base tokens might be removed from ``._vocab_base`` ,
        if they were never found in the training samples. The base vocabulary being changed, ``chr(id)``
        would then bind to incorrect bytes (on which byte succession would not have been learned). We
        register the original id/token/byte association in ``._vocab_base_id_to_byte`` and
        ``._vocab_base_byte_to_token`` .

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
    def special_tokens(self) -> Sequence[str]:
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
            return self._rests_ticks[self._current_midi_metadata["time_division"]][0]

    def preprocess_midi(self, midi: MidiFile):
        r"""Pre-process (in place) a MIDI file to quantize its time and note attributes
        before tokenizing it. Its notes attribute (times, pitches, velocities) will be
        quantized and sorted, duplicated notes removed, as well as tempos. Empty tracks
        (with no note) will be removed from the MIDI object. Notes with pitches outside
        of self.pitch_range will be deleted.

        :param midi: MIDI object to preprocess.
        """
        # Merge instruments of the same program / inst before preprocessing them
        # This allows to avoid potential duplicated notes in some multitrack settings
        if self.config.use_programs and self.one_token_stream:
            merge_same_program_tracks(midi.instruments)

        t = 0
        while t < len(midi.instruments):
            # quantize notes attributes
            self._quantize_notes(midi.instruments[t].notes, midi.ticks_per_beat)
            # sort notes
            midi.instruments[t].notes.sort(key=lambda x: (x.start, x.pitch, x.end))
            # remove possible duplicated notes
            remove_duplicated_notes(midi.instruments[t].notes)
            if len(midi.instruments[t].notes) == 0:
                del midi.instruments[t]
                continue

            # Quantize sustain pedal and pitch bend
            if self.config.use_sustain_pedals:
                self._quantize_sustain_pedals(
                    midi.instruments[t].pedals, midi.ticks_per_beat
                )
            if self.config.use_pitch_bends:
                self._quantize_pitch_bends(
                    midi.instruments[t].pitch_bends, midi.ticks_per_beat
                )
            # TODO quantize control changes
            t += 1

        # Recalculate max_tick is this could have changed after notes quantization
        if len(midi.instruments) > 0:
            midi.max_tick = max(
                [max([note.end for note in track.notes]) for track in midi.instruments]
            )

        if self.config.use_tempos:
            self._quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

        if len(midi.time_signature_changes) == 0:  # can sometimes happen
            midi.time_signature_changes.append(
                TimeSignature(*TIME_SIGNATURE, 0)
            )  # 4/4 by default in this case
        if self.config.use_time_signatures:
            self._quantize_time_signatures(
                midi.time_signature_changes, midi.ticks_per_beat
            )

    def _quantize_notes(self, notes: List[Note], time_division: int):
        r"""Quantize the notes attributes: their pitch, velocity, start and end values.
        It shifts the notes so that they start at times that match the time resolution
        (e.g. 16 samples per bar).
        Note durations will be clipped to the maximum duration that can be handled by the tokenizer. This is done
        to prevent having incorrect offset values when computing rests.
        Notes with pitches outside of self.pitch_range will be deleted.

        :param notes: notes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_sample = int(time_division / max(self.config.beat_res.values()))
        max_duration_ticks = max(tu[1] for tu in self.config.beat_res) * time_division
        i = 0
        pitches = range(*self.config.pitch_range)
        while i < len(notes):
            if notes[i].pitch not in pitches:
                del notes[i]
                continue
            start_offset = notes[i].start % ticks_per_sample
            notes[i].start += (
                -start_offset
                if start_offset <= ticks_per_sample / 2
                else ticks_per_sample - start_offset
            )
            if notes[i].end - notes[i].start > max_duration_ticks:
                notes[i].end = notes[i].start + max_duration_ticks
            else:
                end_offset = notes[i].end % ticks_per_sample
                notes[i].end += (
                    -end_offset
                    if end_offset <= ticks_per_sample / 2
                    else ticks_per_sample - end_offset
                )

                # if this happens to often, consider using a higher beat resolution
                # like 8 samples per beat or 24 samples per bar
                if notes[i].start == notes[i].end:
                    notes[i].end += ticks_per_sample

            notes[i].velocity = int(
                self.velocities[
                    int(np.argmin(np.abs(self.velocities - notes[i].velocity)))
                ]
            )
            i += 1

    def _quantize_tempos(self, tempos: List[TempoChange], time_division: int):
        r"""Quantize the times and tempo values of tempo change events.
        Consecutive identical tempo changes will be removed.

        :param tempos: tempo changes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_sample = int(time_division / max(self.config.beat_res.values()))
        prev_tempo = TempoChange(-1, -1)
        i = 0
        while i < len(tempos):
            # Quantize tempo value
            tempos[i].tempo = self.tempos[
                np.argmin(np.abs(self.tempos - tempos[i].tempo))
            ]
            if (
                self.config.delete_equal_successive_tempo_changes
                and tempos[i].tempo == prev_tempo.tempo
            ):
                del tempos[i]
                continue
            rest = tempos[i].time % ticks_per_sample
            tempos[i].time += (
                -rest if rest <= ticks_per_sample / 2 else ticks_per_sample - rest
            )

            # If the current tempo is now at the same time as the previous one, we delete the previous
            if tempos[i].time == prev_tempo.time:
                prev_tempo = tempos[i]
                del tempos[i - 1]
                continue

            prev_tempo = tempos[i]
            i += 1

    def _quantize_time_signatures(
        self, time_sigs: List[TimeSignature], time_division: int
    ):
        r"""Quantize the time signature changes, delayed to the next bar.
        See MIDI 1.0 Detailed specifications, pages 54 - 56, for more information on
        delayed time signature messages.

        :param time_sigs: time signature changes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_bar = MIDITokenizer._compute_ticks_per_bar(
            time_sigs[0], time_division
        )
        previous_tick = 0  # first time signature change is always at tick 0
        prev_ts = time_sigs[0]
        i = 1
        while i < len(time_sigs):
            time_sig = time_sigs[i]

            if self.config.delete_equal_successive_time_sig_changes and (
                time_sig.numerator,
                time_sig.denominator,
            ) == (prev_ts.numerator, prev_ts.denominator):
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

            # If the current time signature is now at the same time as the previous one, we delete the previous
            if time_sig.time == previous_tick:
                previous_tick = time_sig.time
                del time_sigs[i - 1]
                continue

            previous_tick = time_sig.time
            prev_ts = time_sig
            i += 1

    def _quantize_sustain_pedals(self, pedals: List[Pedal], time_division: int):
        r"""Quantize the sustain pedal events from a track. Their onset and offset times will be adjusted
        according to the beat resolution of the tokenizer.

        :param pedals: sustain pedal events.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_sample = int(time_division / max(self.config.beat_res.values()))
        for pedal in pedals:
            start_offset = pedal.start % ticks_per_sample
            end_offset = pedal.end % ticks_per_sample
            pedal.start += (
                -start_offset
                if start_offset <= ticks_per_sample / 2
                else ticks_per_sample - start_offset
            )
            pedal.end += (
                -end_offset
                if end_offset <= ticks_per_sample / 2
                else ticks_per_sample - end_offset
            )
            if pedal.start == pedal.end:
                pedal.end += ticks_per_sample
            pedal.duration = pedal.end - pedal.start

    def _quantize_pitch_bends(self, pitch_bends: List[PitchBend], time_division: int):
        r"""Quantize the pitch bend events from a track. Their onset and offset times will be adjusted
        according to the beat resolution of the tokenizer.

        :param pitch_bends: pitch bend events.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_sample = int(time_division / max(self.config.beat_res.values()))
        for pitch_bend in pitch_bends:
            start_offset = pitch_bend.time % ticks_per_sample
            pitch_bend.time += (
                -start_offset
                if start_offset <= ticks_per_sample / 2
                else ticks_per_sample - start_offset
            )
            pitch_bend.pitch = self.pitch_bends[
                np.argmin(np.abs(self.pitch_bends - pitch_bend.pitch))
            ]

    def _midi_to_tokens(
        self, midi: MidiFile, *args, **kwargs
    ) -> Union[TokSequence, List[TokSequence]]:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        The workflow of this method is as follows: the events (Pitch, Velocity, Tempo, TimeSignature...) are
        gathered into a list, then the time events are added. If `one_token_stream` is true, all events of all tracks
        are treated all at once, otherwise the events of each track are treated independently.

        :param midi: the MIDI objet to convert.
        :return: a :class:`miditok.TokSequence` if `tokenizer.one_token_stream` is true, else a list of
                :class:`miditok.TokSequence` objects.
        """
        # Create events list
        all_events = []
        if not self.one_token_stream:
            for i in range(len(midi.instruments)):
                all_events.append([])

        # Global events (Tempo, TimeSignature)
        global_events = self._create_midi_events(midi)
        if self.one_token_stream:
            all_events += global_events
        else:
            for i in range(len(all_events)):
                all_events[i] += global_events

        # Adds note tokens
        for ti, track in enumerate(midi.instruments):
            note_events = self._create_track_events(track)
            if self.one_token_stream:
                all_events += note_events
            else:
                all_events[ti] += note_events

        # Add time events
        if self.one_token_stream:
            if (self.config.use_sustain_pedals or self.config.use_pitch_bends) and len(
                midi.instruments
            ) > 1:
                # We also sort by token type here so that they all come in the same order
                all_events.sort(key=lambda x: (x.time, self.__order(x)))
            else:
                all_events.sort(key=lambda x: x.time)
            all_events = self._add_time_events(all_events)
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                all_events[i].sort(key=lambda x: x.time)
                all_events[i] = self._add_time_events(all_events[i])
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

    @staticmethod
    def __order(event: Event) -> int:
        # Global MIDI tokens first
        if event.type in ["Tempo", "TimeSig"]:
            return 0
        # Track effects then
        elif event.type in ["Pedal", "PedalOff"]:
            return 1
        elif event.type == "Program" and event.desc == "ProgramPitchBend":
            return 2
        elif event.type == "PitchBend":
            return 2
        elif event.type == "ControlChange":
            return 3
        # Track notes then
        else:
            return 10

    def _create_track_events(self, track: Instrument) -> List[Event]:
        r"""Extract the tokens / events of individual tracks: `Pitch`, `Velocity`, `Duration`, `NoteOn`, `NoteOff` and
        optionally `Chord`, from a track (``miditoolkit.Instrument``).

        :param track: MIDI track to convert
        :return: sequence of corresponding Events
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        program = track.program if not track.is_drum else -1
        events = []
        note_token_name = "NoteOn" if self._note_on_off else "Pitch"

        # Add chords
        if self.config.use_chords and not track.is_drum:
            chords = detect_chords(
                track.notes,
                self._current_midi_metadata["time_division"],
                chord_maps=self.config.chord_maps,
                specify_root_note=self.config.chord_tokens_with_root_note,
                beat_res=self._first_beat_res,
                unknown_chords_nb_notes_range=self.config.chord_unknown,
            )
            for chord in chords:
                if self.config.use_programs:
                    events.append(
                        Event("Program", track.program, chord.time, "ProgramChord")
                    )
                events.append(chord)

        # Add sustain pedal
        if self.config.use_sustain_pedals:
            for pedal in track.pedals:
                # If not using programs, the default value is 0
                events.append(
                    Event(
                        "Pedal", program if self.config.use_programs else 0, pedal.start
                    )
                )
                # PedalOff or Duration
                if self.config.sustain_pedal_duration:
                    index = np.argmin(np.abs(dur_bins - pedal.duration))
                    events.append(
                        Event(
                            "Duration",
                            ".".join(map(str, self.durations[index])),
                            pedal.start,
                        )
                    )
                else:
                    events.append(Event("PedalOff", program, pedal.end))

        # Add pitch bend
        if self.config.use_pitch_bends:
            for pitch_bend in track.pitch_bends:
                if self.config.use_programs:
                    events.append(
                        Event(
                            "Program",
                            program,
                            pitch_bend.time,
                            "ProgramPitchBend",
                        )
                    )
                events.append(Event("PitchBend", pitch_bend.pitch, pitch_bend.time))

        # TODO add control changes

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Pitch / Velocity
            if self.config.use_programs:
                events.append(
                    Event(type="Program", value=program, time=note.start, desc=note.end)
                )
            events.append(
                Event(
                    type=note_token_name,
                    value=note.pitch,
                    time=note.start,
                    desc=note.end,
                )
            )
            events.append(
                Event(
                    type="Velocity",
                    value=note.velocity,
                    time=note.start,
                    desc=f"{note.velocity}",
                )
            )

            # Duration / NoteOff
            if self._note_on_off:
                if self.config.use_programs:
                    events.append(
                        Event(
                            type="Program", value=program, time=note.end, desc=note.end
                        )
                    )
                events.append(
                    Event(
                        type="NoteOff", value=note.pitch, time=note.end, desc=note.end
                    )
                )
            else:
                duration = note.end - note.start
                index = np.argmin(np.abs(dur_bins - duration))
                events.append(
                    Event(
                        type="Duration",
                        value=".".join(map(str, self.durations[index])),
                        time=note.start,
                        desc=f"{duration} ticks",
                    )
                )

        return events

    def _create_midi_events(self, midi: MidiFile) -> List[Event]:
        r"""Create the *global* MIDI additional tokens: `Tempo` and `TimeSignature`.

        :param midi: midi to extract the events from.
        :return: list of Events.
        """
        events = []

        # Adds tempo events if specified
        if self.config.use_tempos:
            for tempo_change in midi.tempo_changes:
                events.append(
                    Event(
                        type="Tempo",
                        value=tempo_change.tempo,
                        time=tempo_change.time,
                        desc=tempo_change.tempo,
                    )
                )

        # Add time signature tokens if specified
        if self.config.use_time_signatures:
            for time_signature_change in midi.time_signature_changes:
                events.append(
                    Event(
                        type="TimeSig",
                        value=f"{time_signature_change.numerator}/{time_signature_change.denominator}",
                        time=time_signature_change.time,
                    )
                )

        return events

    def _add_time_events(self, events: List[Event]) -> List[Event]:
        raise NotImplementedError

    def midi_to_tokens(
        self, midi: MidiFile, apply_bpe_if_possible: bool = True, *args, **kwargs
    ) -> Union[TokSequence, List[TokSequence]]:
        r"""Tokenizes a MIDI file.
        This method returns a list of :class:`miditok.TokSequence`.

        If you are implementing your own tokenization by subclassing this class, **override the
        ``_midi_to_tokens`` method**. This method implement necessary MIDI preprocessing.

        :param midi: the MIDI object to convert.
        :param apply_bpe_if_possible: will apply BPE if the tokenizer's vocabulary was learned with.
        :return: a :class:`miditok.TokSequence` if `tokenizer.one_token_stream` is true, else a list of
                :class:`miditok.TokSequence` objects.
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self._durations_ticks:
            self._durations_ticks[midi.ticks_per_beat] = np.array(
                [
                    (beat * res + pos) * midi.ticks_per_beat // res
                    for beat, pos, res in self.durations
                ]
            )
        if self.config.use_rests:
            if midi.ticks_per_beat not in self._rests_ticks:
                self._rests_ticks[midi.ticks_per_beat] = np.array(
                    [
                        (beat * res + pos) * midi.ticks_per_beat // res
                        for beat, pos, res in self.rests
                    ]
                )

        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self._current_midi_metadata = {
            "time_division": midi.ticks_per_beat,
            "max_tick": midi.max_tick,
            "tempo_changes": midi.tempo_changes,
            "time_sig_changes": midi.time_signature_changes,
            "key_sig_changes": midi.key_signature_changes,
        }

        tokens = self._midi_to_tokens(midi, *args, **kwargs)

        if apply_bpe_if_possible and self.has_bpe:
            self.apply_bpe(tokens)

        return tokens

    def complete_sequence(self, seq: TokSequence):
        r"""Completes (inplace) a :class:`miditok.TokSequence` object by converting its attributes.
        The input sequence can miss some of its attributes (ids, tokens), but needs at least one for reference.
        This method will create the missing ones from the present ones.
        The ``bytes`` attribute will be created if the tokenizer has been trained with BPE.
        The ``events`` attribute will not be filled as it is only intended for debug purpose.

        :param seq: input :class:`miditok.TokSequence`, must have at least one attribute defined.
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

        if self.has_bpe:
            if seq.bytes is None:
                seq.bytes = self._ids_to_bytes(seq.ids, as_one_str=True)

    def _tokens_to_ids(
        self, tokens: Sequence[Union[str, List[str]]]
    ) -> List[Union[int, List[int]]]:
        r"""Converts a list of tokens (str) into their associated ids (int).

        :param tokens: list of tokens (str) to convert.
        :return: list of corresponding ids (int).
        """
        if isinstance(tokens[0], (list, tuple)):
            ids = []
            for seq in tokens:
                ids.append([self[i, token] for i, token in enumerate(seq)])
        else:
            ids = [self[token] for token in tokens]
        return ids

    def _ids_to_tokens(
        self, ids: List[Union[int, List[int]]], as_str: bool = True
    ) -> List[Union[Union[str, Event], List[Union[str, Event]]]]:
        r"""Converts a sequence of ids (int) to their associated tokens (str or Event).
        **This method will not work with ids encoded with BPE. You will need to decode them
        first (:py:meth:`miditok.MIDITokenizer.decode_bpe`).**

        :param ids: sequence of ids (int) to convert.
        :param as_str: return the tokens as string objects, otherwise Event objects (default: True)
        :return: the sequence of corresponding tokens (str or Event).
        """
        tokens = []
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
        events: List[Union[Event, List[Event]]]
    ) -> List[Union[str, List[str]]]:
        r"""Converts a sequence of Events to their associated tokens (str).

        :param events: sequence of Events to convert.
        :return: the sequence of corresponding tokens (str).
        """
        tokens = []
        if isinstance(events[0], list):  # multiple vocabularies
            for (
                multi_event
            ) in events:  # cannot use recursion here because of the vocabulary type id
                multi_token = []
                for i, event in enumerate(multi_event):
                    multi_token.append(str(event))
                tokens.append(multi_token)
            return tokens

        tokens = [str(event) for event in events]
        return tokens

    def _ids_to_bytes(
        self, ids: List[Union[int, List[int]]], as_one_str: bool = False
    ) -> Union[str, List[str]]:
        r"""Converts a list of ids into their associated bytes.
        It can be returned either as a list of bytes or as a unique string of bytes.
        **This method will not work with ids encoded with BPE. You will need to decode them
        first (:py:meth:`miditok.MIDITokenizer.decode_bpe`).**

        :param ids: token ids (int) to convert.
        :param as_one_str: will return the bytes all concatenated into one string. (default: False)
        :return: the tokens converted into strings of unique bytes.
        """
        if isinstance(ids[0], list):
            return [self._ids_to_bytes(item, as_one_str) for item in ids]
        bytes_ = [self._vocab_base_id_to_byte[i] for i in ids]
        return "".join(bytes_) if as_one_str else bytes_

    def _bytes_to_tokens(
        self, bytes_: Union[str, List[str]], as_str: bool = True
    ) -> List[Union[Union[str, Event], List[Union[str, Event]]]]:
        r"""Converts a sequence of bytes into their associated tokens (str or Event).

        :param bytes_: sequence of bytes to convert.
        :param as_str: return the events as string objects, otherwise Event objects (default: True)
        :return: the sequence of corresponding tokens (str).
        """
        if isinstance(bytes_[0], list):  # multiple vocabularies
            return [self._bytes_to_tokens(byte_) for byte_ in bytes_]

        tokens = []
        for byte_ in bytes_:
            token_str = self._vocab_bpe_bytes_to_tokens[byte_]
            tokens.append(token_str if as_str else Event(*token_str.split("_")))
        tokens = [tok for toks in tokens for tok in toks]  # flatten
        return tokens

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        programs: Optional[List[Tuple[int, bool]]] = None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = TIME_DIVISION,
    ) -> MidiFile:
        r"""Detokenize one or multiple sequences of tokens into a MIDI file.
        You can give the tokens sequences either as :class:`miditok.TokSequence` objects, lists of integers,
        numpy arrays or PyTorch / Tensorflow tensors.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
                a Tensor (PyTorch and Tensorflow are supported), a numpy array or a Python list of ints.
                The first dimension represents tracks, unless the tokenizer handle tracks altogether as a
                single token sequence (e.g. Octuple, MuMIDI): tokenizer.one_token_stream == True.
        :param programs: programs of the tracks. If none is given, will default to piano, program 0. (default: None)
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (miditoolkit.MidiFile).
        """
        raise NotImplementedError

    @abstractmethod
    def _create_base_vocabulary(self, *args, **kwargs) -> List[Union[str, List[str]]]:
        r"""Creates the vocabulary, as a list of string tokens.
        This method is unimplemented and need to be overridden by inheriting classes.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Do not include special tokens. These have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        raise NotImplementedError

    def __create_vocabulary(self):
        r"""Method actually creating the vocabulary object, as Dictionary, from the ``_create_vocabulary``
        method implemented by tokenization classes.
        This method is called at ``__init__``\.
        """
        vocab = self._create_base_vocabulary()
        special_tokens = [f"{tok}_None" for tok in self.special_tokens]

        if isinstance(vocab[0], list):  # multi-voc
            self._vocab_base = [{} for _ in range(len(vocab))]
            self.__vocab_base_inv = [{} for _ in range(len(vocab))]
            for vid in range(len(vocab)):
                vocab[vid] = special_tokens + vocab[vid]
                for tok in vocab[vid]:
                    self.add_to_vocab(tok, vid)
        else:
            vocab = special_tokens + vocab
            for tok in vocab:
                self.add_to_vocab(tok)

    def _add_additional_tokens_to_vocab_list(self, vocab: List[str]):
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

        def create_for_dict(voc: Dict[str, int]):
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

    def token_ids_of_type(self, token_type: str, vocab_id: int = None) -> List[int]:
        r"""Returns the list of token ids of the given type.

        :param token_type: token type to get the associated token ids.
        :param vocab_id: index of the vocabulary associated to the token, if applicable. (default: None)
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
        vocab_idx: int = None,
        byte_: str = None,
        add_to_bpe_model: bool = False,
    ):
        r"""Adds an event to the vocabulary. Its index (int) will be the length of the vocab.

        :param token: token to add, as a formatted string of the form "Type_Value", e.g. Pitch_80, or an Event.
        :param vocab_idx: idx of the vocabulary (in case of embedding pooling). (default: None)
        :param byte_: unique byte associated to the token. This is used when building the vocabulary with
            fast BPE. If None is given, it will default to ``chr(id_ + CHR_ID_START)`` . (default: None)
        :param add_to_bpe_model: the token will be added to the bpe_model vocabulary too. (default: None)
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
        """Just create the *Chord* tokens that will populate the base vocabulary.
        This protected method is intended to be used by subclasses when creating their vocabularies.

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

    def token_id_type(self, id_: int, vocab_id: int = None) -> str:
        r"""Returns the type of the given token id.

        :param id_: token id to get the type.
        :param vocab_id: index of the vocabulary associated to the token, if applicable. (default: None)
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
        Two exceptions are made for the special BOS (Beginning of Sequence) and EOS (End of Sequence) tokens:
        No token type can precede a BOS token, and EOS token cannot precede any other token.
        """
        original_token_types = list(self.tokens_types_graph.keys())
        for special_token in self.config.special_tokens:
            if special_token == "EOS":
                self.tokens_types_graph["EOS"] = []
            else:
                self.tokens_types_graph[special_token] = original_token_types + list(
                    self.config.special_tokens
                )

            if special_token != "BOS":
                for token_type in original_token_types:
                    self.tokens_types_graph[token_type].append(special_token)

    def __create_durations_tuples(self) -> List[Tuple[int, int, int]]:
        r"""Creates the possible durations in beat / position units, as tuple of the form:
        (beat, pos, res) where beat is the number of beats, pos the number of "samples"
        and res the beat resolution considered (samples per beat).
        Example: (2, 5, 8) means the duration is 2 beat long + position 5 / 8 of the ongoing beat
        In pure ticks we have: duration = (beat * res + pos) * time_division // res
            Is equivalent to: duration = nb_of_samples * ticks_per_sample
        So in the last example, if time_division is 384: duration = (2 * 8 + 5) * 384 // 8 = 1008 ticks

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
        r"""Converts a *Duration* token value of the form x.x.x, for beat.position.resolution,
        in ticks. Can also be used for *TimeShift* tokens.

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
        time_division: int = None,
        rest: bool = False,
    ) -> Tuple[List[Tuple[int, int, int]], List[int]]:
        r"""Converts a duration in ticks into a sequence of `Duration` / `TimeShift` values.

        :param duration: duration in tick to convert.
        :param time_division: time division of the MIDI being parsed. If none is given, the method will use
            `self._current_midi_metadata["time_division"]`. (default: None)
        :param rest: the duration is a rest, hence the created tokens will be based on the `self.rests` values.
        :return: list of associated token values, and the list of the elapsed offset in tick for each of these values.
        """
        if time_division is None:
            time_division = self._current_midi_metadata["time_division"]
        if rest:
            dur_bins = self._rests_ticks[time_division]
            dur_vals = self.rests
        else:
            dur_bins = self._durations_ticks[time_division]
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
        using either log or linear scaled values based on the value of self.config.log_tempos.

        :return: the tempos.
        """
        tempo_fn = np.geomspace if self.config.log_tempos else np.linspace
        tempos = tempo_fn(*self.config.tempo_range, self.config.nb_tempos).round(2)

        return tempos

    def __create_time_signatures(self) -> List[Tuple]:
        r"""Creates the possible time signatures, as tuples of the form:
        (nb_beats, beat_res) where nb_beats is the number of beats per bar.
        Example: (3, 4) means one bar is 3 beat long and each beat is a quarter note.

        :return: the time signatures.
        """
        time_signature_range = self.config.time_signature_range

        time_signatures = []
        for beat_res, beats in time_signature_range.items():
            assert (
                beat_res > 0 and math.log2(beat_res).is_integer()
            ), f"The beat resolution ({beat_res}) in time signature must be a power of 2"

            time_signatures.extend([(nb_beats, beat_res) for nb_beats in beats])

        return time_signatures

    def __create_pitch_bends(self) -> np.ndarray:
        r"""Creates the pitch bend values, as numpy array, using `self.config.pitch_bend_range`.

        :return: the pitch bend values.
        """
        return np.linspace(*self.config.pitch_bend_range, dtype=np.int32)

    @staticmethod
    def _compute_ticks_per_bar(time_sig: TimeSignature, time_division: int):
        r"""Computes time resolution of one bar in ticks.

        :param time_sig: time signature object
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :return: MIDI bar resolution, in ticks/bar
        """
        return int(time_division * 4 * time_sig.numerator / time_sig.denominator)

    @staticmethod
    def _parse_token_time_signature(token_time_sig: str) -> Tuple[int, int]:
        r"""Converts a time signature token value of the form x/x into a tuple of integers,
        time signature's numerator (bar length in beats) and denominator (beat resolution).

        :param token_time_sig: TimeSig token value.
        :return: the numerator and denominator of a time signature.
        """
        numerator, denominator = map(int, token_time_sig.split("/"))
        return numerator, denominator

    def validate_midi_time_signatures(self, midi: MidiFile) -> bool:
        r"""Checks if MIDI files contains only time signatures supported by the encoding.
        :param midi: MIDI file
        :return: boolean indicating whether MIDI file could be processed by the Encoding
        """
        if self.config.use_time_signatures:
            for time_sig in midi.time_signature_changes:
                if (
                    time_sig.numerator,
                    time_sig.denominator,
                ) not in self.time_signatures:
                    return False
        return True

    def learn_bpe(
        self,
        vocab_size: int,
        iterator: Iterable = None,
        tokens_paths: List[Union[Path, str]] = None,
        start_from_empty_voc: bool = False,
        **kwargs,
    ):
        r"""Method to construct the vocabulary from BPE, backed by the 🤗tokenizers library.
        The data used for training can either be given through the ``iterator`` argument as
        an iterable object yielding strings, or by ``tokens_paths`` as a list of paths to
        token json files that will be loaded.
        You can read the Hugging Face `🤗tokenizers documentation
        <https://huggingface.co/docs/tokenizers/training_from_memory>`_,
        `🤗tokenizers API documentation <https://huggingface.co/docs/tokenizers/python/v0.9.4/api/reference.html#>`_
        and `🤗tokenizers course <https://huggingface.co/course/chapter6/2?fw=pt>`_
        for more details about the ``iterator`` and input type.

        **The training progress bar will not appear with non-proper terminals.**
        (cf `GitHub issue <https://github.com/huggingface/tokenizers/issues/157>`_ )

        :param vocab_size: size of the vocabulary to learn / build.
        :param iterator: an iterable object yielding the training data, as lists of string.
            It can be a list or a Generator. This iterator will be passed to the BPE model for
            training. If None is given, you must use the ``tokens_paths`` argument. (default: None)
        :param tokens_paths: paths of the token json files to load and use. (default: False)
        :param start_from_empty_voc: the training will start from an empty base vocabulary.
            The tokenizer will then have a base vocabulary only based on the unique bytes present
            in the training data. If you set this argument to True, you should use the tokenizer only
            with the training data, as new data might contain "unknown" tokens missing from the vocabulary.
            Comparing this to text, setting this argument to True would create a tokenizer that will only know
            the characters present in the training data, and would not be compatible / know other characters.
            This argument can allow to optimize the vocabulary size.
            If you are unsure about this, leave it to False. (default: False)
        :param kwargs: any additional argument to pass to the trainer.
        """
        if self.is_multi_voc:
            print(
                "This tokenizer is based on multiple vocabularies / embedding pooling. It is therefore not compatible"
                "with Byte Pair Encoding (BPE). Skipping this function call (learn_bpe)."
            )
            return
        assert (
            iterator is not None or tokens_paths is not None
        ), "You must give at an iterator or a path to to token "

        if vocab_size <= len(self.vocab):
            print(
                f"vocab_size ({vocab_size}) need to be higher than the size of the current vocabulary "
                f"({len(self.vocab)}). Skipping BPE training."
            )
            return

        # If no iterator, loads tokens / samples to analyze
        if iterator is None:
            iterator = []  # list of lists of one string (bytes)
            for file_path in tqdm(tokens_paths, desc="Loading token files"):
                sample = self.load_tokens(file_path)
                bytes_ = self._ids_to_bytes(
                    sample["ids"], as_one_str=True
                )  # list of str (bytes)
                iterator += (
                    [[byte_] for byte_ in bytes_]
                    if not self.one_token_stream
                    else [bytes_]
                )

            # This doesn't seem to work, the trainer pre-processes the sequences, but then no word remains
            """def it_gen():
                for file_path_ in tqdm(tokens_paths, desc="Loading token files"):
                    sample_ = self.load_tokens(file_path_)
                    yield self._ids_to_bytes(sample_["ids"], as_one_str=True)

            iterator = it_gen()"""

            # Make sure the target vocab size > nb of unique chars across all samples
            unique_chars = set()
            for seq in iterator:
                unique_chars.update(*seq)

            if len(unique_chars) >= vocab_size:
                print(
                    f"BPE TRAINING: the provided data comprises {len(unique_chars)} base tokens (character level), "
                    f"whereas the target BPE vocabulary size is inferior ({vocab_size}). No new token can be learned, "
                    f"skipping BPE training."
                )
                return

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
        special_tokens_bytes = self._ids_to_bytes(
            self._tokens_to_ids([f"{tok}_None" for tok in self.config.special_tokens])
        )
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens_bytes,
            show_progress=True,
            **kwargs,
        )
        self._bpe_model.train_from_iterator(
            iterator, length=sum(1 for _ in iterator), trainer=trainer
        )

        # Update other vocabs accordingly
        if start_from_empty_voc:
            # If we do not give an existing vocabulary to the tokenizer, 🤗tokenizers first fill its
            # vocabulary with all bytes present in the training samples, sorted by byte / char index.
            # Some bytes / tokens might be missing from tokenizer.get_vocab(), as simply not
            # present in training samples. We must get rid of them from the base vocabulary
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
            for (
                byte_,
                id_,
            ) in (
                new_vocab.items()
            ):  # dict is ordered so id val is incremented each time, from 0
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

    def apply_bpe_to_dataset(
        self, dataset_path: Union[Path, str], out_path: Union[Path, str] = None
    ):
        r"""Applies BPE to an already tokenized dataset (with no BPE).

        :param dataset_path: path to token files to load.
        :param out_path: output directory to save. If none is given, this method will overwrite original files.
                (default: None)
        """
        if not self.has_bpe:
            return

        files_paths = list(Path(dataset_path).glob("**/*.json"))
        for path in tqdm(files_paths, desc="Applying BPE to dataset"):
            sample = self.load_tokens(path)
            seq = (
                TokSequence(ids=sample["ids"])
                if self.one_token_stream
                else [TokSequence(ids=track) for track in sample["ids"]]
            )
            self.apply_bpe(seq)

            out_ = (
                Path(out_path) / path.relative_to(dataset_path)
                if out_path is not None
                else path
            )
            self.save_tokens(
                seq, out_, sample["programs"] if "programs" in sample else None
            )

    def _are_ids_bpe_encoded(self, ids: Union[List[int], np.ndarray]) -> bool:
        r"""A small check telling if a sequence of ids are encoded with BPE.
        This is performed by checking if any id has a value superior or equal to the length
        of the base vocabulary.

        :param ids: ids to check
        :return: boolean, True if ids are encoded with BPE, False otherwise.
        """
        return np.any(np.array(ids) >= len(self._vocab_base))

    def decode_bpe(self, seq: Union[TokSequence, List[TokSequence]]):
        r"""Decodes (inplace) a sequence of tokens (:class:`miditok.TokSequence`) with ids encoded with BPE.
        This method only modifies the ``.ids`` attribute of the input sequence(s) only and does not complete it.
        This method can also receive a list of sequences, in which case it will decompose BPE on each of them
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
        midi_paths: Union[List[str], List[Path]],
        out_dir: Union[str, Path],
        tokenizer_config_file_name: str = "tokenizer.conf",
        validation_fn: Callable[[MidiFile], bool] = None,
        data_augment_offsets=None,
        apply_bpe: bool = True,
        save_programs: bool = None,
        logging: bool = True,
    ):
        r"""Converts a dataset / list of MIDI files, into their token version and save them as json files
        The resulting Json files will have the shape (T, *), first dimension is tracks, second tokens.
        In order to reduce disk space usage, **only the ids are saved**.
        If save_programs is True, the shape will be [(T, *), (T, 2)], first dim is tokens and programs instead,
        for programs the first value is the program, second a bool indicating if the track is drums.
        The config of the tokenizer will be saved as a "config.txt" file by default.

        :param midi_paths: paths of the MIDI files.
        :param out_dir: output directory to save the converted files.
        :param tokenizer_config_file_name: name of the tokenizer config file name. This file will be saved in
            `out_dir`. (default: "tokenizer.conf")
        :param validation_fn: a function checking if the MIDI is valid on your requirements
            (e.g. time signature, minimum/maximum length, instruments ...).
        :param data_augment_offsets: data augmentation arguments, to be passed to the
            miditok.data_augmentation.data_augmentation_dataset method. Has to be given as a list / tuple
            of offsets pitch octaves, velocities, durations, and finally their directions (up/down). (default: None)
        :param apply_bpe: will apply BPE on the dataset to save, if the vocabulary was learned with. (default: True)
        :param save_programs: will save the programs of the tracks of the MIDI as an entry in the Json file.
            That this option is probably unnecessary when using a multitrack tokenizer (`config.use_programs`), as the
            Program information is present within the tokens, and that the tracks having the same programs are likely
            to have been merged. (default: False if `config.use_programs`, else True)
        :param logging: logs progress bar.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Saves the tokenizer so that it can be reloaded
        self.save_params(out_dir / tokenizer_config_file_name)
        if save_programs is None:
            save_programs = not self.config.use_programs

        for midi_path in (
            tqdm(
                midi_paths,
                desc=f'Tokenizing MIDIs ({"/".join(list(out_dir.parts[-2:]))})',
            )
            if logging
            else midi_paths
        ):
            # Some MIDIs can contain errors that are raised by Mido, if so the loop continues
            midi_path = Path(midi_path)
            try:
                midi = MidiFile(midi_path)
            except FileNotFoundError:
                if logging:
                    print(f"File not found: {midi_path}")
                continue
            except Exception:
                # known are ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            # Checks the time division is valid
            if midi.ticks_per_beat < max(self.config.beat_res.values()) * 4:
                continue
            # Passing the MIDI to validation tests if given
            if validation_fn is not None:
                if not validation_fn(midi):
                    continue

            # Checks if MIDI contains supported time signatures
            if not self.validate_midi_time_signatures(midi):
                continue

            # Tokenizing the MIDI, without BPE here as this will be done at the end (as we might perform data aug)
            tokens = self(midi, apply_bpe_if_possible=False)

            # Save the tokens as JSON
            self.save_tokens(
                tokens,
                Path(out_dir, f"{Path(midi_path).stem}.json").with_suffix(".json"),
                get_midi_programs(midi) if save_programs else None,
            )

        # Perform data augmentation
        if data_augment_offsets is not None:
            data_augmentation_dataset(out_dir, self, *data_augment_offsets)

        if apply_bpe and self.has_bpe:
            self.apply_bpe_to_dataset(out_dir)

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(
        self, tokens: Union[TokSequence, List[Union[int, List[int]]]]
    ) -> Union[float, List[float]]:
        r"""Checks if a sequence of tokens is made of good token types
        successions and returns the error ratio (lower is better).
        The common implementation in MIDITokenizer class will check token types,
        duplicated notes and time errors. It works for REMI, TSD and Structured.
        Other tokenizations override this method to include other errors
        (like no NoteOff / NoteOn for MIDILike and embedding pooling).
        Overridden methods must call decompose_bpe at the beginning if BPE is used!

        :param tokens: sequence of tokens to check.
        :return: the error ratio (lower is better).
        """
        # If list of TokSequence -> recursive
        if isinstance(tokens, list):
            return [self.tokens_errors(tok_seq) for tok_seq in tokens]

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
        note_tokens_types = ["Pitch", "NoteOn"]

        # Init first note and current pitches if needed
        if previous_type in note_tokens_types:
            pitch_val = int(tokens[0].split("_")[1])
            current_pitches[current_program].append(pitch_val)
        elif previous_type == "Position":
            current_pos = int(tokens[0].split("_")[1])

        for ti, token in enumerate(tokens[1:]):
            # err_tokens = tokens[ti - 4 : ti + 4]  # uncomment for debug
            event_type, event_value = token.split("_")[0], token.split("_")[1]

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type == "Bar":  # reset
                    current_pos = -1
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in ["TimeShift", "Time-Shift", "Rest"]:
                    current_pitches = {p: [] for p in self.config.programs}
                elif event_type in note_tokens_types:
                    pitch_val = int(event_value)
                    if pitch_val in current_pitches[current_program]:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches[current_program].append(pitch_val)
                elif event_type == "Position":
                    if int(event_value) <= current_pos and previous_type != "Rest":
                        err_time += 1  # token position value <= to the current position
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
        programs: List[Tuple[int, bool]] = None,
        **kwargs,
    ):
        r"""Saves tokens as a JSON file.
        In order to reduce disk space usage, **only the ids are saved**.
        Use kwargs to save any additional information within the JSON file.

        :param tokens: tokens, as list, numpy array, torch or tensorflow Tensor.
        :param path: path of the file to save.
        :param programs: (optional), programs of the associated tokens, should be
                        given as a tuples (int, bool) for (program, is_drum).
        :param kwargs: any additional information to save within the JSON file.
        """
        ids = []
        ids_bpe_encoded = None

        if isinstance(tokens, TokSequence):
            self.complete_sequence(tokens)
            ids_bpe_encoded = tokens.ids_bpe_encoded
            ids = tokens.ids
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

        with open(path, "w") as outfile:
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
        with open(path) as file:
            return json.load(file)

    def save_params(
        self, out_path: Union[str, Path], additional_attributes: Dict = None
    ):
        r"""Saves the config / parameters of the tokenizer in a json encoded file.
        This can be useful to keep track of how a dataset has been tokenized.
        **Note:** if you override this method, you should probably call it (super()) at the end
        and use the additional_attributes argument.

        :param out_path: output path to save the file.
        :param additional_attributes: any additional information to store in the config file.
                It can be used to override the default attributes saved in the parent method. (default: None)
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
            "miditok_version": CURRENT_VERSION_PACKAGE,
            **additional_attributes,
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as outfile:
            json.dump(params, outfile, indent=4)

    def _load_params(self, config_file_path: Union[str, Path]):
        r"""Loads the parameters of the tokenizer from a config file.
        This method is not intended to be called outside __init__, when creating a tokenizer.

        :param config_file_path: path to the tokenizer config file (encoded as json).
        """
        with open(config_file_path) as param_file:
            params = json.load(param_file)

        # Grab config, or creates one with default parameters (for retro-compatibility with previous version)
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
        r"""Calling a tokenizer allows to directly convert a MIDI to tokens or the other way around.
        The method automatically detects MIDI and token objects, as well as paths and can directly load
        MIDI or token json files before converting them.
        This will call the :py:func:`miditok.MIDITokenizer.midi_to_tokens` if you provide a MIDI object
        or path to a MIDI file, or the :py:func:`miditok.MIDITokenizer.tokens_to_midi` method otherwise.

        :param obj: a `miditoolkit.MidiFile` object, a sequence of tokens, or a path to a MIDI or tokens json file.
        :return: the converted object.
        """
        # Tokenize MIDI
        if isinstance(obj, MidiFile):
            return self.midi_to_tokens(obj, *args, **kwargs)

        # Loads a file (.mid or .json)
        elif isinstance(obj, (str, Path)):
            path = Path(obj)
            if path.suffix in MIDI_FILES_EXTENSIONS:
                midi = MidiFile(obj)
                return self.midi_to_tokens(midi, *args, **kwargs)
            else:
                tokens = self.load_tokens(path)
                return self.tokens_to_midi(tokens, *args, **kwargs)

        # Consider it tokens --> converts to MIDI
        else:
            return self.tokens_to_midi(obj, *args, **kwargs)

    def __len__(self) -> int:
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling / have multiple vocabularies, it will return the **sum** of their lengths.
        If the vocabulary was learned with fast BPE, it will return the length of the BPE vocabulary,
        i.e. the proper number of possible token ids. Otherwise, it will return the length of the base vocabulary.
        Use the :py:func:`miditok.MIDITokenizer.len` property (``tokenizer.len``) to have the list of lengths.

        :return: length of the vocabulary.
        """
        if self.is_multi_voc:
            return sum([len(v) for v in self.vocab])
        elif self.has_bpe:
            return len(self._bpe_model.get_vocab())
        return len(self.vocab)

    @property
    def len(self) -> Union[int, List[int]]:
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling / have multiple vocabularies, it will return the **list** of their lengths.
        Use the :py:func:`miditok.MIDITokenizer.__len__` magic method (``len(tokenizer)``)
        to get the sum of the lengths.

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

        :param item: a token (int) or an event (str). For tokenizers with embedding pooling / multiple vocabularies
            ( `tokenizer.is_multi_voc` ), you must either provide a string (token) that is within all vocabularies (e.g.
            special tokens), or a tuple where the first element in the index of the vocabulary and the second the
            element to index.
        :return: the converted object.
        """
        if isinstance(item, tuple) and self.is_multi_voc:
            return self.__get_from_voc(item[1], item[0])
        elif self.is_multi_voc and isinstance(item, str):
            if all(item in voc for voc in self.vocab):
                return [voc[item] for voc in self.vocab]
            else:
                raise ValueError(
                    "This tokenizer uses multiple vocabularies / embedding pooling. To index it you must"
                    "either provide a token (string) that is within all vocabularies (e.g. special tokens), or a tuple"
                    "where the first element in the index of the vocabulary and the second the element to index."
                )
        else:
            return self.__get_from_voc(item)

    def __get_from_voc(
        self, item: Union[int, str], vocab_id: int = None
    ) -> Union[int, str]:
        r"""Get element from the vocabulary.
        The method handles both token (int) <--> event (str) ways.

        :param item: item to get / index.
        :param vocab_id: index of the vocabulary associated to the token, if applicable. (default: None)
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
        r"""Checks if two tokenizers are identical. This is done by comparing their vocabularies,
        and configuration.

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
