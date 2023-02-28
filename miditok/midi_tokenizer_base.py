"""
MIDI encoding base class and methods
"""
from abc import ABC, abstractmethod
import math
from pathlib import Path
import json
from random import choices
from copy import deepcopy
from typing import List, Tuple, Dict, Union, Callable, Iterable, Optional, Any

import numpy as np
from tqdm import tqdm
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .classes import Event, TokSequence
from .utils import remove_duplicated_notes, get_midi_programs, convert_ids_tensors_to_list
from .data_augmentation import data_augmentation_dataset
from .constants import TIME_DIVISION, CURRENT_VERSION_PACKAGE, PITCH_RANGE, BEAT_RES, NB_VELOCITIES, \
    ADDITIONAL_TOKENS, SPECIAL_TOKENS, CHR_ID_START


def _in_as_seq(complete: bool = True, decode_bpe: bool = True):
    """Decorator creating if necessary and completing a Sequence object before that the function is called.
    ``track_to_tokens``
    :param complete: will complete the sequence.
    :param decode_bpe: will decode BPE, if applicable. This step is performed before completing the sequence.
    """
    def decorator(function: Callable = None):
        def wrapper(*args, **kwargs):
            self = args[0]
            seq = args[1]
            if not isinstance(seq, TokSequence):
                ids = tokens = events = None
                if isinstance(seq[0], int) or (isinstance(seq[0], list) and isinstance(seq[0][0], int)):
                    ids = convert_ids_tensors_to_list(seq)
                elif isinstance(seq[0], str) or (isinstance(seq[0], str) and isinstance(seq[0][0], str)):
                    tokens = seq
                else:  # list of Event, very unlikely
                    events = seq

                seq = TokSequence(ids=ids, tokens=tokens, events=events)

            if self.has_bpe and decode_bpe:
                self.decompose_bpe(seq)
            if complete:
                self.complete_sequence(seq)

            args = list(args)
            args[1] = seq
            return function(*args, **kwargs)
        return wrapper
    return decorator


def _out_as_complete_seq(function: Callable):
    """Decorator completing a output Sequence object.
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        res = function(*args, **kwargs)
        self.complete_sequence(res)
        return res

    return wrapper


class MIDITokenizer(ABC):
    r"""MIDI tokenizer base class, containing common methods and attributes for all tokenizers.

    :param pitch_range: (default: range(21, 109)) range of MIDI pitches to use. Pitches can take
            values between 0 and 127 (included).
            The `General MIDI 2 (GM2) specifications <https://www.midi.org/specifications-old/item/general-midi-2>`_
            indicate the **recommended** ranges of pitches per MIDI program (instrument).
            These recommended ranges can also be found in ``miditok.constants``.
            In all cases, the range from 21 to 108 (included) covers all the recommended values.
            When processing a MIDI, the notes with pitches under or above this range can be discarded.
    :param beat_res: (default: `{(0, 4): 8, (4, 12): 4}`) beat resolutions, as a dictionary in the form:
            ``{(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}``
            The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar, and
            the values are the resolution (in samples per beat) to apply to the ranges, ex 8.
            This allows to use **Duration** / **TimeShift** tokens of different lengths / resolutions.
            Note: for tokenization with **Position** tokens, the total number of possible positions will
            be set at four times the maximum resolution given (``max(beat_res.values)``).
    :param nb_velocities: (default: 32) number of velocity bins. In the MIDI norm, velocities can take
            up to 128 values (0 to 127). This parameter allows to reduce the number of velocity values.
            The velocities of the MIDIs resolution will be downsampled to ``nb_velocities`` values, equally
            separated between 0 and 127.
    :param additional_tokens: (default: None used) specify which additional tokens to use.
            Compatibilities between tokenization and additiona tokens may vary.
            See :ref:`Additional tokens` for the details and available tokens.
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "BOS", "EOS", "MASK"]``)
    :param unique_track: set to True if the tokenizer works only with a unique track.
            Tokens will be saved as a single track. This applies to representations that natively handle
            multiple tracks such as Octuple, resulting in a single "stream" of tokens for all tracks.
            This attribute will be saved in config files of the tokenizer. (default: False)
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int, Tuple[int, int]]] = ADDITIONAL_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS,
        unique_track: bool = False,
        params: Union[str, Path] = None,
    ):
        # Initialize params
        self._vocab_base = {}  # vocab of prime tokens, can be viewed as unique char / bytes
        self.__vocab_base_inv = {}  # the other way, to decode id (int) -> token (str)
        self._vocab_base_id_to_byte = {}  # id (int) -> byte (str), as this might not be chr(id) after BPE training
        self._vocab_base_byte_to_token = {}  # byte (str) -> token (str), for basic tokens
        self._vocab_bpe_bytes_to_tokens = {}  # byte(s) -> token(s)
        self.has_bpe = False
        self.__bpe_slow = False  # to route to slow BPE methods
        if special_tokens is not None:
            special_tokens = [f"{tok}_None" for tok in special_tokens]
        else:
            special_tokens = []

        # Fast BPE tokenizer backed with 🤗tokenizers
        if not self.is_multi_voc:
            self._bpe_model = TokenizerFast(
                BPE(
                    vocab={},
                    merges=[],
                    dropout=None,
                    continuing_subword_prefix="",
                    end_of_word_suffix="",
                    fuse_unk=False,
                )
            )
        else:
            self._bpe_model = None

        # Loading params, or
        if params is not None:
            self.load_params(params)
        else:
            assert pitch_range.start >= 0 and pitch_range.stop <= 128, \
                "You must specify a pitch_range between 0 and 127 (included, i.e. range.stop at 128)"
            assert 0 < nb_velocities < 128,\
                "You must specify a nb_velocities between 0 and 127 (included)"
            self._pitch_range = pitch_range
            self._beat_res = beat_res
            self._nb_velocities = nb_velocities
            self.additional_tokens = additional_tokens
            self.special_tokens = special_tokens
            self.unique_track = unique_track

        # Init duration and velocity values
        self._durations = self.__create_durations_tuples()
        self._velocities = np.linspace(0, 127, self._nb_velocities + 1, dtype=np.intc)[
            1:
                           ]  # remove velocity 0
        self._first_beat_res = list(self._beat_res.values())[0]
        for beat_range, res in self._beat_res.items():
            if 0 in beat_range:
                self._first_beat_res = res
                break

        # Tempos
        self._tempos = np.zeros(1)
        if self.additional_tokens["Tempo"]:
            self._tempos = np.linspace(
                *self.additional_tokens["tempo_range"],
                self.additional_tokens["nb_tempos"],
                dtype=np.intc,
            )

        # Rests
        self._rests = []
        if self.additional_tokens["Rest"]:
            assert (
                self.additional_tokens["rest_range"][0] // 4 <= self._first_beat_res
            ), "The minimum rest value must be equal or superior to the initial beat resolution"
            self._rests = self.__create_rests()

        # Time Signatures
        self._time_signatures = []
        if self.additional_tokens["TimeSignature"]:
            self._time_signatures = self.__create_time_signatures()

        # Vocabulary and token types graph
        if len(self.vocab) == 0:  # in case it was already loaded by an overridden load_params method, such as with BPE
            self.__create_vocabulary()
            self._vocab_base_id_to_byte = {i: chr(i + CHR_ID_START) for i in range(len(self._vocab_base))}
            self._vocab_base_byte_to_token = {chr(i + CHR_ID_START): ev for ev, i in self._vocab_base.items()}
        self.tokens_types_graph = self._create_token_types_graph()
        if "BOS" in special_tokens or "EOS" in special_tokens:
            self._add_bos_eos_to_types_graph()

        # Slow BPE attributes
        self.__bpe_successions = {}
        if self.__bpe_slow:  # loaded from config file TODO retrieve slow_bpe attribute from loading params, or property ?
            self.__set_bpe_slow_tokens_successions()

        # Keep in memory durations in ticks for seen time divisions so these values
        # are not calculated each time a MIDI is processed
        self._durations_ticks = {}

        # Holds the tempo changes, time signature, time division and key signature of a
        # MIDI (being parsed) so that methods processing tracks can access them
        self._current_midi_metadata = {}  # needs to be updated each time a MIDI is read

    @property
    def vocab(self) -> Union[Dict[str, int], List[Dict[str, int]]]:  # token (str) to its id (int)
        """
        Token: ``str`` describing a unique event, e.g. *Pitch_50*
        Id: ``int`` the associated integer index of a token, to be fed to a model.

        ``._vocab_base``: Dict[str: int] token -> id - Registers all known base tokens.
        ``.__vocab_base_inv``: Dict[int: str] id -> token - Inverse of ``._base_vocab``, to go the other way.
        ``._vocab_bpe_bytes_to_tokens``: Dict[str: List[str]] byte(s) -> token(s) used to decode BPE encoded sequences.
        ``._bpe_model.get_vocab()``: Dict[str: int] byte -> id - bpe model vocabulary, based on unique bytes

        Token to byte is achieved by doing ``char(self._vocab_base[token])``, basically we convert its id to a byte.

        :return:
        """
        return self._vocab_base

    def preprocess_midi(self, midi: MidiFile):
        r"""Pre-process (in place) a MIDI file to quantize its time and note attributes
        before tokenizing it. Its notes attribute (times, pitches, velocities) will be
        quantized and sorted, duplicated notes removed, as well as tempos. Empty tracks
        (with no note) will be removed from the MIDI object. Notes with pitches outside
        of self.pitch_range will be deleted.

        :param midi: MIDI object to preprocess.
        """
        t = 0
        while t < len(midi.instruments):
            self._quantize_notes(
                midi.instruments[t].notes, midi.ticks_per_beat
            )  # quantize notes attributes
            midi.instruments[t].notes.sort(
                key=lambda x: (x.start, x.pitch, x.end)
            )  # sort notes
            remove_duplicated_notes(
                midi.instruments[t].notes
            )  # remove possible duplicated notes
            if len(midi.instruments[t].notes) == 0:
                del midi.instruments[t]
                continue
            t += 1

        # Recalculate max_tick is this could have change after notes quantization
        if len(midi.instruments) > 0:
            midi.max_tick = max(
                [max([note.end for note in track.notes]) for track in midi.instruments]
            )

        if self.additional_tokens["Tempo"]:
            self._quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

        if len(midi.time_signature_changes) == 0:  # can sometimes happen
            midi.time_signature_changes.append(
                TimeSignature(4, 4, 0)
            )  # 4/4 by default in this case
        if self.additional_tokens["TimeSignature"]:
            self._quantize_time_signatures(
                midi.time_signature_changes, midi.ticks_per_beat
            )

    def _quantize_notes(
        self, notes: List[Note], time_division: int
    ):
        r"""Quantize the notes attributes: their pitch, velocity, start and end values.
        It shifts the notes so that they start at times that match the time resolution
        (e.g. 16 samples per bar).
        Notes with pitches outside of self.pitch_range will be deleted.

        :param notes: notes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_sample = int(time_division / max(self._beat_res.values()))
        i = 0
        while i < len(notes):
            if notes[i].pitch not in self._pitch_range:
                del notes[i]
                continue
            start_offset = notes[i].start % ticks_per_sample
            end_offset = notes[i].end % ticks_per_sample
            notes[i].start += (
                -start_offset
                if start_offset <= ticks_per_sample / 2
                else ticks_per_sample - start_offset
            )
            notes[i].end += (
                -end_offset
                if end_offset <= ticks_per_sample / 2
                else ticks_per_sample - end_offset
            )

            if (
                notes[i].start == notes[i].end
            ):  # if this happens to often, consider using a higher beat resolution
                notes[
                    i
                ].end += (
                    ticks_per_sample  # like 8 samples per beat or 24 samples per bar
                )

            notes[i].velocity = int(
                self._velocities[
                    int(np.argmin(np.abs(self._velocities - notes[i].velocity)))
                ]
            )
            i += 1

    def _quantize_tempos(self, tempos: List[TempoChange], time_division: int):
        r"""Quantize the times and tempo values of tempo change events.
        Consecutive identical tempo changes will be removed.

        :param tempos: tempo changes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_sample = int(time_division / max(self._beat_res.values()))
        prev_tempo = -1
        i = 0
        while i < len(tempos):
            # Quantize tempo value
            tempos[i].tempo = self._tempos[
                np.argmin(np.abs(self._tempos - tempos[i].tempo))
            ]
            if tempos[i].tempo == prev_tempo:
                del tempos[i]
                continue
            rest = tempos[i].time % ticks_per_sample
            tempos[i].time += (
                -rest if rest <= ticks_per_sample / 2 else ticks_per_sample - rest
            )
            prev_tempo = tempos[i].tempo
            i += 1

    @staticmethod
    def _quantize_time_signatures(time_sigs: List[TimeSignature], time_division: int):
        r"""Quantize the time signature changes, delayed to the next bar.
        See MIDI 1.0 Detailed specifications, pages 54 - 56, for more information on
        delayed time signature messages.

        :param time_sigs: time signature changes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        """
        ticks_per_bar = time_division * time_sigs[0].numerator
        current_bar = 0
        previous_tick = 0  # first time signature change is always at tick 0
        prev_time_sig = time_sigs[0]
        i = 1
        while i < len(time_sigs):
            time_sig = time_sigs[i]

            if (time_sig.numerator, time_sig.denominator) == (
                prev_time_sig.numerator,
                prev_time_sig.denominator,
            ) or time_sig.time == previous_tick:
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
            ticks_per_bar = time_division * time_sig.numerator
            current_bar += bar_offset
            previous_tick = time_sig.time
            prev_time_sig = time_sig
            i += 1

    def _midi_to_tokens(
        self, midi: MidiFile, *args, **kwargs
    ) -> List[TokSequence]:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        Tokenization treating all tracks as a single token sequence might
        override this method, e.g. Octuple or PopMAG.

        :param midi: the MIDI objet to convert.
        :return: sequences of tokens.
        """
        # Convert each track to tokens
        tokens = []
        for track in midi.instruments:
            tokens.append(self.track_to_tokens(track))
            self.complete_sequence(tokens[-1])
        return tokens

    def midi_to_tokens(
        self, midi: MidiFile, apply_bpe_if_possible: bool = True, *args, **kwargs
    ) -> List[TokSequence]:
        r"""Tokenize a MIDI file.
        **Override the ``_midi_to_tokens`` method if needed. This method implement necessary MIDI preprocessing.**

        :param midi: the MIDI objet to convert.
        :param apply_bpe_if_possible: will apply BPE if the tokenizer has learn it.
        :return: sequences of tokens.
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self._durations_ticks:
            self._durations_ticks[midi.ticks_per_beat] = np.array(
                [
                    (beat * res + pos) * midi.ticks_per_beat // res
                    for beat, pos, res in self._durations
                ]
            )

        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self._current_midi_metadata = {
            "time_division": midi.ticks_per_beat,
            "tempo_changes": midi.tempo_changes,
            "time_sig_changes": midi.time_signature_changes,
            "key_sig_changes": midi.key_signature_changes,
        }

        tokens = self._midi_to_tokens(midi, *args, **kwargs)

        if apply_bpe_if_possible and self.has_bpe:
            self.apply_bpe(tokens)

        return tokens

    @abstractmethod
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens.
        This method is unimplemented and need to be overridden by inheriting classes.
        For an easier implementation, use the _out_as_complete_seq decorator.

        :param track: MIDI track to convert.
        :return: sequence of corresponding tokens.
        """
        raise NotImplementedError

    def complete_sequence(self, seq: TokSequence):
        r"""Completes (inplace) a Sequence object by converting its attributes.
        The input sequence can miss some of its attributes (ids, tokens), but needs at least one for reference.
        This method will create the missing ones from the present ones.
        The ``bytes`` attribute will be created if the tokenizer has been trained with BPE.
        The ``events`` attribute will not be filled as it is only intended for debug purpose.

        :param seq: input sequence, must have one defined.
        """
        if seq.tokens is None:
            if seq.events is not None:
                seq.tokens = [str(event) for event in seq.events]
            elif seq.ids is not None:
                seq.tokens = self._ids_to_tokens(seq.ids)
            elif seq.bytes is not None:
                seq.tokens = self._bytes_to_tokens(seq.bytes)
        if seq.ids is None:
            seq.ids = self._tokens_to_ids(seq.tokens)

        if self.has_bpe:  # TODO handle the case where ids are bpe encoded with vocab
            if seq.bytes is None:
                seq.bytes = self._ids_to_bytes(seq.ids, as_one_str=True)

    def _tokens_to_ids(self, tokens: List[Union[str, List[str]]]) -> List[Union[int, List[int]]]:
        r"""Converts a list of Event objects into a list of tokens.
        It will apply BPE if it has been learned.

        :param tokens: list of tokens (str) to convert.
        :return: list of corresponding tokens.
        """
        if isinstance(tokens[0], list):
            ids = []
            for seq in tokens:
                ids.append([self[i, token] for i, token in enumerate(seq)])
        else:
            ids = [self[token] for token in tokens]
        return ids

    def _ids_to_tokens(
        self, ids: List[Union[int, List[int]]], as_str: bool = True
    ) -> List[Union[Union[str, Event], List[Union[str, Event]]]]:
        r"""Convert a sequence of tokens in their respective event objects.
        BPE tokens will be decoded.

        :param ids: sequence of tokens to convert.
        :param as_str: return the tokens as string objects, otherwise Event objects (default: True)
        :return: the sequence of corresponding tokens (str).
        """
        tokens = []
        if isinstance(ids[0], list):  # multiple vocabularies
            for multi_ids in ids:  # cannot use recursion here because of the vocabulary type id
                multi_event = []
                for i, token in enumerate(multi_ids):
                    event_str = self[i, token]
                    multi_event.append(event_str if as_str else Event(*event_str.split("_")))
                tokens.append(multi_event)
            return tokens

        ids_ = self.decompose_bpe(ids) if self.has_bpe else ids
        for id_ in ids_:
            event_str = self[id_]
            tokens.append(event_str if as_str else Event(*event_str.split("_")))
        return tokens

    def _ids_to_bytes(self, ids: List[Union[int, List[int]]], as_one_str: bool = False) -> Union[str, List[str]]:
        r"""Converts a list of ids into a string of unique bytes.

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
        r"""Convert a sequence of tokens in their respective event objects.

        :param bytes_: sequence of tokens to convert.
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

    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        programs: Optional[List[Tuple[int, bool]]] = None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = TIME_DIVISION,
    ) -> MidiFile:
        r"""Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        **NOTE:** With Remi, MIDI-Like, CP Word or other encoding methods that process tracks
        independently, only the tempo changes of the first track in tokens will be used

        :param tokens: tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a list of TokSequences.
                The first dimension represents tracks, unless the tokenizer handle tracks altogether as a
                single token sequence (e.g. Octuple, MuMIDI): tokenizer.unique_track == True.
        :param programs: programs of the tracks.
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (miditoolkit.MidiFile).
        """
        midi = MidiFile(ticks_per_beat=time_division)
        for i, track_tokens in enumerate(tokens):
            if programs is not None:
                track, tempo_changes = self.tokens_to_track(
                    track_tokens, time_division, programs[i]
                )
            else:
                track, tempo_changes = self.tokens_to_track(track_tokens, time_division)
            midi.instruments.append(track)
            if i == 0:  # only keep tempo changes of the first track
                midi.tempo_changes = tempo_changes
                midi.tempo_changes[0].time = 0
        midi.max_tick = max(
            [
                max([note.end for note in track.notes]) if len(track.notes) > 0 else 0
                for track in midi.instruments
            ]
        )

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    @abstractmethod
    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object.
        This method is unimplemented and need to be overridden by inheriting classes.
        This method should be decorated with _in_as_complete_seq to receive any type of input.

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :param program: the MIDI program of the produced track and if it drum. (default (0, False), piano)
        :return: the miditoolkit instrument object and the possible tempo changes.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_base_vocabulary(self, *args, **kwargs) -> List[Union[str, List[str]]]:
        r"""Creates the vocabulary, as a list of string events.
        This method is unimplemented and need to be overridden by inheriting classes.
        Each event as to be given as the form of "Type_Value", separated with an underscore.
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
        This method is called at ``__init__``.
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

    def add_to_vocab(self, event: Union[str, Event], vocab_idx: int = None):
        r"""Adds an event to the vocabulary. Its index (int) will be the length of the vocab.

        :param event: event to add, as a formatted string of the form "Type_Value", e.g. Pitch_80
        :param vocab_idx: idx of the vocabulary (in case of embedding pooling). (default: None)
        """
        event_str = event if isinstance(event, str) else str(event)  # TODO handle all vocabs

        if vocab_idx is not None:
            self._vocab_base[vocab_idx][event_str] = len(self._vocab_base[vocab_idx])
            self.__vocab_base_inv[vocab_idx][len(self.__vocab_base_inv[vocab_idx])] = event_str
        else:
            self.vocab[event_str] = len(self.vocab)
            self.__vocab_base_inv[len(self.__vocab_base_inv)] = event_str

    def token_type(self, id_: int, vocab_id: int = None) -> str:
        r"""Returns the type of the given token.

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

    def _add_bos_eos_to_types_graph(self):
        r"""Adds (inplace) special tokens (*EOS* and *EOS* only) types
        to the token types graph dictionary.
        """
        self.tokens_types_graph["SOS"] = list(self.tokens_types_graph.keys())
        self.tokens_types_graph["EOS"] = []
        for value in self.tokens_types_graph.values():
            value.append("EOS")

    def __create_durations_tuples(self) -> List[Tuple]:
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
        for beat_range, beat_res in self._beat_res.items():
            durations += [
                (beat, pos, beat_res)
                for beat in range(*beat_range)
                for pos in range(beat_res)
            ]
        durations += [
            (max(max(self._beat_res)), 0, self._beat_res[max(self._beat_res)])
        ]  # the last one
        del durations[0]  # removes duration of 0
        return durations

    @staticmethod
    def _token_duration_to_ticks(token_duration: str, time_division: int) -> int:
        r"""Converts a *Duration* token value of the form x.x.x, for beat.position.resolution,
        in ticks. Can also be used for *TimeShift* tokens.

        :param token_duration: Duration / TimeShift token value
        :param time_division: time division
        :return: the duration / time-shift in ticks
        """
        beat, pos, res = map(int, token_duration.split("."))
        return (beat * res + pos) * time_division // res

    def __create_rests(self) -> List[Tuple]:
        r"""Creates the possible rests in beat / position units, as tuple of the form:
        (beat, pos) where beat is the number of beats, pos the number of "samples"
        The rests are calculated from the value of self.additional_tokens[rest_range],
        which first value divide a beat to determine the minimum rest represented,
        and the second the maximum rest in beats.
        The rests shorter than 1 beat will scale x2, as rests in music theory (semiquaver, quaver, crotchet...)
        Note that the values of the rests in positions will be determined by the beat
        resolution of the first range (self.beat_res)

        Example: (4, 6) and a first beat resolution of 8 will give the rests:
            [(0, 2), (0, 4), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]

        :return: the rests.
        """
        div, max_beat = self.additional_tokens["rest_range"]
        assert (
            div % 2 == 0 and div <= self._first_beat_res
        ), f"The minimum rest must be divisible by 2 and lower than the first beat resolution ({self._first_beat_res})"
        rests = []
        while div > 1:
            rests.append((0, self._first_beat_res // div))
            div //= 2
        rests += [(i, 0) for i in range(1, max_beat + 1)]
        return rests

    def __create_time_signatures(self) -> List[Tuple]:
        r"""Creates the possible time signatures, as tuple of the form:
        (nb_beats, beat_res) where nb_beats is the number of beats per bar.
        Example: (3, 4) means one bar is 3 beat long and each beat is a quarter note.

        :return: the time signatures.
        """
        max_beat_res, nb_notes = self.additional_tokens.get(
            "time_signature_range", (4, 1)
        )
        assert (
            max_beat_res > 0 and math.log2(max_beat_res).is_integer()
        ), "The beat resolution in time signature must be a power of 2"

        time_signatures = []
        for i in range(0, int(math.log2(max_beat_res)) + 1):  # 1 ~ max_beat_res
            for j in range(1, ((2**i) * nb_notes) + 1):
                time_signatures.append((j, 2**i))
        return time_signatures

    def _reduce_time_signature(
        self, numerator: int, denominator: int
    ) -> Tuple[int, int]:
        r"""Reduces and decomposes a time signature into one of the valid vocabulary time signatures.
        If time signature's denominator (beat resolution) is larger than max_beat_res,
        the denominator and numerator are reduced to max_beat_res if possible.
        If time signature's numerator (bar length in beats) is larger than nb_notes * denominator,
        the numerator is replaced with its GCD not larger than nb_notes * denominator.

        Example: (10, 4), max_beat_res of 8, and nb_notes of 2 will convert the signature into (5, 4).

        :param numerator: time signature's numerator (bar length in beats).
        :param denominator: time signature's denominator (beat resolution).
        :return: the numerator and denominator of a reduced and decomposed time signature.
        """
        max_beat_res, nb_notes = self.additional_tokens["time_signature_range"]

        # reduction (when denominator exceed max_beat_res)
        while (
            denominator > max_beat_res and denominator % 2 == 0 and numerator % 2 == 0
        ):
            denominator //= 2
            numerator //= 2

        assert denominator <= max_beat_res, (
            f"Unsupported time signature ({numerator}/{denominator}), "
            f"beat resolution is irreducible to maximum beat resolution {max_beat_res}"
        )

        # decomposition (when length of a bar exceed max_nb_beats_per_bar)
        while numerator > nb_notes * denominator:
            for i in range(2, numerator + 1):
                if numerator % i == 0:
                    numerator //= i
                    break

        return numerator, denominator

    @staticmethod
    def _parse_token_time_signature(token_time_sig: str) -> Tuple[int, int]:
        r"""Converts a time signature token value of the form x/x into a tuple of integers,
        time signature's numerator (bar length in beats) and denominator (beat resolution).

        :param token_time_sig: TimeSig token value.
        :return: the numerator and denominator of a time signature.
        """
        numerator, denominator = map(int, token_time_sig.split("/"))
        return numerator, denominator

    def learn_bpe(
            self,
            vocab_size: int,
            tokens_paths: List[Union[Path, str]] = None,
            iterator: Iterable = None,
            files_lim: int = None,
            load_all_token_files_once: bool = False,
            start_from_empty_voc: bool = False,
            **kwargs
    ):
        """Method to construct the vocabulary from BPE, backed by the 🤗tokenizers library.
        The data used for training can either be given through the ``iterator`` argument as
        as iterable object yielding strings, or by ``tokens_paths`` as a list of paths to
        token json files that will be loaded.

        :param vocab_size: size of the vocabulary to learn / build.
        :param tokens_paths: paths of the token json files to load and use (default: False)
        :param iterator: (default: False)
        :param files_lim: (default: False)
        :param load_all_token_files_once: if using ``tokens_paths``, all the tokens files will
                be loaded once at the beginning and kept in memory, else they will be loaded
                on the fly during training. (default: False)
        :param start_from_empty_voc: (default: False)
        :param kwargs: any additional argument to pass to the trainer.
        """
        assert iterator is not None or tokens_paths is not None, \
            "You must give at an iterator or a path to to token "

        # Create new tokenizer model
        tokenizer_json = json.loads(self._bpe_model.to_str())
        nb_bytes = len(self.special_tokens) if start_from_empty_voc else len(self._vocab_base)
        voc_start = {chr(i + CHR_ID_START): i for i in range(nb_bytes)}
        tokenizer_json["model"]["vocab"] = voc_start  # byte (str) -> id (int)
        tokenizer_json["model"]["merges"] = []
        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))

        # Get files paths
        if files_lim is not None and files_lim < len(tokens_paths):
            tokens_paths = choices(tokens_paths, k=files_lim)

        # If no iterator, loads tokens / samples to analyze
        if iterator is None:
            if load_all_token_files_once:
                samples = []  # list of lists of one string (bytes)
                for file_path in tqdm(tokens_paths, desc="Loading token files"):
                    sample = self.load_tokens(file_path)
                    bytes_ = self._ids_to_bytes(sample["ids"], as_one_str=True)  # list of str (bytes)
                    samples += [[byte_] for byte_ in bytes_]
                iterator = samples

            else:
                def iterator():
                    for file_path_ in tqdm(tokens_paths, desc="Loading token files"):
                        sample_ = self.load_tokens(file_path_)
                        yield self._ids_to_bytes(sample_["ids"], as_one_str=True)

                iterator = iterator()

        # Loads tokens / samples to analyze
        """samples = []
        for file_path in tqdm(tokens_paths, desc="Loading token files"):
            sample = self.load_tokens(file_path)
            samples.append(self.__ids_to_bytes(sample["ids"], as_one_str=True))
        iterator = samples"""

        # Trains the tokenizer
        special_tokens_bytes = self._ids_to_bytes(self._tokens_to_ids(self.special_tokens))
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens_bytes, show_progress=True, **kwargs)
        tokenizer.train_from_iterator(iterator, length=sum(1 for _ in iterator), trainer=trainer)

        # Update other vocabs accordingly
        if start_from_empty_voc:
            # If we do not give an existing vocabulary to the tokenizer, 🤗tokenizers first fill its
            # vocabulary with all bytes present in the training samples, sorted by byte / char index.
            # Some bytes / tokens might be missing from tokenizer.get_vocab(), as simply not
            # present in training samples. We must get rid of them from the base vocabulary
            new_vocab = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}  # byte -> id
            byte_to_token_old = deepcopy(self._vocab_base_byte_to_token)

            # Rebuild _base_vocab
            self._vocab_base = {}  # token -> id
            self.__vocab_base_inv = {}  # id -> token
            self._vocab_base_byte_to_token = {}  # for all basic tokens
            self._vocab_base_id_to_byte = {}
            for byte_, id_ in new_vocab.items():  # dict is ordered so id val is incremented each time, from 0
                if byte_ in byte_to_token_old:
                    token = byte_to_token_old[byte_]  # get the original token associated to the byte
                    self.add_to_vocab(token)  # adds it to _vocab_base
                    self._vocab_base_byte_to_token[byte_] = token
                    self._vocab_base_id_to_byte[id_] = byte_

        # Update __vocab_bpe_bytes_to_tokens for faster decoding
        self._vocab_bpe_bytes_to_tokens = {k: [self._vocab_base_byte_to_token[b] for b in k]
                                           for k in tokenizer.get_vocab()}

        self._bpe_model = tokenizer
        self.has_bpe = True

    def apply_bpe(self, seq: Union[TokSequence, List[TokSequence]]) -> Union[TokSequence, List[TokSequence]]:

        if self.__bpe_slow:  # will call slow encoding method, one seq at a time
            if isinstance(seq, list):
                for seq_ in seq:
                    seq_.ids = self.__apply_bpe_slow(seq_.ids)
                    seq_.ids_bpe_encoded = True
            else:
                seq.ids = self.__apply_bpe_slow(seq.ids)
                seq.ids_bpe_encoded = True

        elif isinstance(seq, list):  # TODO batch it
            for seq_ in seq:
                self.complete_sequence(seq_)
            '''encoded_tokens2 = self._bpe_model.encode_batch([t.bytes for t in seq], is_pretokenized=True).ids
            for seq, bpe_tokens in zip(encoded_tokens, encoded_tokens):
                seq.ids = bpe_tokens'''

        else:
            self.complete_sequence(seq)
            encoded_tokens = self._bpe_model.encode([seq.bytes], is_pretokenized=True)
            seq.ids = encoded_tokens.ids
            seq.ids_bpe_encoded = True

    def learn_bpe_slow(
        self,
        tokens_path: Union[Path, str],
        vocab_size: int,
        out_dir: Union[Path, str],
        files_lim: int = None,
        save_converted_samples: bool = False,
        print_seq_len_variation: bool = True,
    ) -> Tuple[List[float], List[int], List[float]]:
        r"""Byte Pair Encoding (BPE) method to build the vocabulary.
        This method will build (modify) the vocabulary by analyzing an already tokenized dataset to find
        the most recurrent token successions.
        Note that this implementation is in pure Python and will be slow if you use a large amount of
        tokens files. You might use the files_lim argument.

        :param tokens_path: path to token files to learn the BPE combinations from.
        :param vocab_size: the new vocabulary size.
        :param out_dir: directory to save the tokenizer's parameters and vocabulary after BPE learning is finished.
        :param files_lim: limit of token files to use. (default: None)
        :param save_converted_samples: will save in out_path the samples that have been used
                to create the BPE vocab. Files will keep the same name and relative path. (default: True)
        :param print_seq_len_variation: prints the mean sequence length before and after BPE,
                and the variation in %. (default: True)
        :return: learning metrics, as lists of:
                - the average number of token combinations covered by the newly created BPE tokens
                - the maximum number of token combinations
                - the average sequence length
                Each index in the list correspond to a learning step.
        """
        assert not self.is_multi_voc, (
            "You are using a multi-vocabulary tokenizer, "
            "it is not compatible with byte pair encoding"
        )
        assert vocab_size > len(self.vocab), (
            f"vocab_size ({vocab_size}) need to be higher than the size"
            f"of the current vocabulary ({len(self.vocab)})"
        )
        if isinstance(out_dir, str):
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files_paths = list(Path(tokens_path).glob("**/*.json"))
        assert (
            len(files_paths) > 0
        ), "BPE learning: the specified path does not contain tokens files (json)"
        files_paths_bpe = (
            choices(files_paths, k=files_lim)
            if (files_lim is not None and files_lim < len(files_paths))
            else files_paths
        )
        samples, samples_paths = [], []
        original_lengths = []

        # Loads tokens / samples to analyze
        for file_path in tqdm(files_paths_bpe, desc="Loading token files"):
            file = self.load_tokens(file_path)
            samples.append(file)
            samples_paths.append(file_path.relative_to(tokens_path))
            original_lengths += (
                [len(file["ids"])]
                if self.unique_track
                else [len(track) for track in file["ids"]]
            )

        def replace_token_in_seq(
            seq: List[int], succession: Tuple[int, int], new_event: str
        ):
            j = 0
            while j < len(seq) - 1:
                if tuple(seq[j: j + 2]) == succession:
                    seq[j] = self[f"BPE_{new_event}"]
                    del seq[j + 1]
                j += 1

        # Learning Byte Pair Encoding
        avg_seq_len = [sum(original_lengths) / len(original_lengths)]
        bpe_comb_nb, bpe_comb_means, bpe_comb_max = [], [], []
        pbar = tqdm(
            total=vocab_size - len(self.vocab), desc="Learning byte pair encoding"
        )
        while len(self.vocab) < vocab_size:
            occurrences = {}  # count occurrences of successive tokens
            for sample in samples:
                tracks = [sample["ids"]] if self.unique_track else sample["ids"]
                for track in tracks:
                    for i in range(len(track) - 1):
                        try:
                            occurrences[tuple(track[i: i + 2])] += 1
                        except KeyError:
                            occurrences[tuple(track[i: i + 2])] = 1

            # Add new BPE token to vocabulary
            most_rec_tok_succession = max(
                occurrences, key=occurrences.get
            )  # most recurrent succession of two tokens
            prime_tokens_eq = []  # the equivalent succession with decomposed BPE tokens
            for token in most_rec_tok_succession:
                if self[token].split("_")[0] == "BPE":
                    prime_tokens_eq += map(
                        int,
                        self[token]
                        .split("_")[1]
                        .split(".")[1]
                        .split("-"),
                    )
                else:
                    prime_tokens_eq.append(token)
            final_event_val = (
                "-".join(map(str, most_rec_tok_succession))
                + "."
                + "-".join(map(str, prime_tokens_eq))
            )
            self.add_to_vocab(f"BPE_{final_event_val}")

            # Replace newly created token in learning samples
            for sample in samples:
                if self.unique_track:
                    replace_token_in_seq(
                        sample["ids"], most_rec_tok_succession, final_event_val
                    )
                else:
                    for track in sample["ids"]:
                        replace_token_in_seq(
                            track, most_rec_tok_succession, final_event_val
                        )

            # Compute metrics
            avg = []
            for sample in samples:
                if self.unique_track:
                    avg.append(len(sample["ids"]))
                else:
                    avg += [len(track) for track in sample["ids"]]
            avg_seq_len.append(np.mean(np.array(avg)).item(0))
            nb_combs = np.array([len(prime_tokens_eq)])  # bpe-combs.prime-combs
            bpe_comb_nb = (
                np.concatenate([bpe_comb_nb, nb_combs])
                if isinstance(bpe_comb_nb, np.ndarray)
                else nb_combs
            )
            bpe_comb_means.append(np.mean(bpe_comb_nb).item(0))
            bpe_comb_max.append(np.max(bpe_comb_nb).item(0))
            pbar.set_postfix(
                {
                    "seq_len_variation": f"{(avg_seq_len[-1] - avg_seq_len[0]) / avg_seq_len[0] * 100:.2f}",
                    "avg_nb_token_combs": f"{bpe_comb_means[-1]:.2f}",
                    "max_nb_token_combs": f"{bpe_comb_max[-1]}",
                },
                refresh=False,
            )
            pbar.update(1)

        # Saves dictionary and prints the difference in sequence length
        pbar.close()
        self.has_bpe = True
        self.__bpe_slow = True
        self.__set_bpe_slow_tokens_successions()
        if save_converted_samples:
            for sample, path in zip(samples, samples_paths):
                self.save_tokens(
                    sample["ids"],
                    Path(out_dir, path).with_suffix(".json"),
                    sample["programs"],
                )
        if print_seq_len_variation:
            print(
                f"Mean of original lengths: {avg_seq_len[0]}\nMean length after BPE: {avg_seq_len[-1]}"
            )
            print(
                f"Variation from original: {(avg_seq_len[-1] - avg_seq_len[0]) / avg_seq_len[0] * 100:.2f} %"
            )
        self.save_params(
            out_dir / "config.txt"
        )  # Saves the parameters with which the MIDIs are converted

        return bpe_comb_means, bpe_comb_max, avg_seq_len

    def __set_bpe_slow_tokens_successions(self):
        r"""For slow BPE.
        Creates the bpe_successions attributes, as a dictionary of the form {bpe_token: (tok1, tok2, tok3...)}
        """
        self.__bpe_successions = {
            tok: list(
                map(
                    int,
                    self[tok]
                    .split("_")[1]
                    .split(".")[0]
                    .split("-"),
                )
            )
            for tok, event in enumerate(self.vocab) if event.split("_")[0] == "BPE"
        }

    def __apply_bpe_slow(self, ids: List[int]) -> List[int]:
        r"""Converts a sequence of tokens into tokens with BPE.

        :param ids: token ids to encode.
        :return: the tokens with BPE applied.
        """
        if not self.has_bpe:
            return ids

        previous_len = (
            len(ids) + 1
        )  # + 1 to fool when entering the loop the first time
        while previous_len != len(
            ids
        ):  # if this is True, it means no more BPE combinations is possible
            previous_len = len(
                ids
            )  # length of the token sequence before applying BPE
            for (
                tok,
                token_succession,
            ) in (
                self.__bpe_successions.items()
            ):  # loops over BPE tokens from the vocabulary
                occurrences = self.__find_subseq(ids, token_succession)
                for idx in reversed(occurrences):
                    ids[idx] = tok
                    for _ in range(len(token_succession) - 1):
                        del ids[idx + 1]
        return ids

    @staticmethod
    def __find_subseq(in_list: List[int], pattern: List[int]) -> List[int]:
        """Finds the locations of a pattern within a list.
        Adapted from: https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
        Related: https://www.reddit.com/r/learnpython/comments/2xqlwj/using_npwhere_to_find_subarrays/
        After testing, the numba jit version does not seem to be much faster.
        The conversion of python lists to numba.typed.List() seems to also take time.

        :param in_list: input list to analyze.
        :param pattern: pattern to detect.
        :return: indices of in_list where the pattern has been found.
        """
        matches = []
        next_possible_idx = 0
        for i in range(len(in_list)):
            if in_list[i] == pattern[0] and in_list[i:i + len(pattern)] == pattern and i >= next_possible_idx:
                matches.append(i)
                next_possible_idx = i + len(pattern)

        return matches

    def apply_bpe_to_dataset(
        self, dataset_path: Union[Path, str], out_path: Union[Path, str]
    ):
        r"""Applies BPE to an already tokenized dataset (with no BPE).

        :param dataset_path: path to token files to load.
        :param out_path: output directory to save.
        """
        if not self.has_bpe:
            return

        files_paths = list(Path(dataset_path).glob("**/*.json"))
        for path in tqdm(files_paths, desc="Applying BPE to dataset"):
            sample = self.load_tokens(path)
            sample_bpe = (
                self.apply_bpe(sample["ids"])
                if self.unique_track
                else [self.apply_bpe(track) for track in sample["ids"]]  # TODO batch it
            )
            self.save_tokens(
                sample_bpe,
                Path(out_path) / path.relative_to(dataset_path),
                sample["programs"],
            )

    def decompose_bpe(self, tokens: List[int]) -> List[int]:
        r"""Decomposes a sequence of tokens containing BP encoded tokens into "prime" tokens.
        It is an inplace operation. TODO for fast bpe
        # TODO adapt calls

        :param tokens: token sequence to decompose.
        :return: decomposed token sequence.
        """

        """
        decoded_events = [tokenizer.vocab_bpe_bytes_to_events[tok] for tok in encoded_tokens.tokens]
        decoded_events = [item for sublist in decoded_events for item in sublist]
        """
        tokens = deepcopy(tokens)
        i = 0
        while i < len(tokens):
            token_type, token_val = self[tokens[i]].split("_")
            if token_type == "BPE":
                del tokens[i]
                for j, to_insert in enumerate(
                    map(int, token_val.split(".")[1].split("-"))
                ):
                    tokens.insert(i + j, to_insert)
            i += 1
        return tokens

    def tokenize_midi_dataset(
        self,
        midi_paths: Union[List[str], List[Path]],
        out_dir: Union[str, Path],
        validation_fn: Callable[[MidiFile], bool] = None,
        data_augment_offsets=None,
        save_programs: bool = True,
        logging: bool = True,
    ):
        r"""Converts a dataset / list of MIDI files, into their token version and save them as json files
        The resulting Json files will have the shape (T, *), first dimension is tracks, second tokens.
        If save_programs is True, the shape will be [(T, *), (T, 2)], first dim is tokens and programs instead,
        for programs the first value is the program, second a bool indicating if the track is drums.
        The config of the tokenizer will be saved as a "config.txt" file by default.

        :param midi_paths: paths of the MIDI files.
        :param out_dir: output directory to save the converted files.
        :param validation_fn: a function checking if the MIDI is valid on your requirements
                            (e.g. time signature, minimum/maximum length, instruments ...).
        :param data_augment_offsets: data augmentation arguments, to be passed to the
            miditok.data_augmentation.data_augmentation_dataset method. Has to be given as a list / tuple
            of offsets pitch octaves, velocities, durations, and finally their directions (up/down). (default: None)
        :param save_programs: will also save the programs of the tracks of the MIDI. (default: True)
        :param logging: logs progress bar.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.save_params(
            out_dir / "config.txt"
        )  # Saves the parameters with which the MIDIs are converted

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
            except (
                Exception
            ):  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            # Checks the time division is valid
            if midi.ticks_per_beat < max(self._beat_res.values()) * 4:
                continue
            # Passing the MIDI to validation tests if given
            if validation_fn is not None:
                if not validation_fn(midi):
                    continue

            # Converting the MIDI to tokens and saving them as json
            tokens = self(midi)
            self.save_tokens(
                tokens,
                Path(out_dir, f"{Path(midi_path).stem}.json").with_suffix(".json"),
                get_midi_programs(midi) if save_programs else None,
            )

        # Perform data augmentation
        if data_augment_offsets is not None:
            data_augmentation_dataset(out_dir, self, *data_augment_offsets)

    @_in_as_seq(complete=False, decode_bpe=False)
    def token_types_errors(self, tokens: Union[TokSequence, List[Union[int, List[int]]]]) -> float:
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
        nb_tok_predicted = len(tokens)  # used to norm the score
        if self.has_bpe:
            self.decompose_bpe(tokens)
        self.complete_sequence(tokens)

        # Override from here
        tokens = tokens.tokens

        err_type = 0  # i.e. incompatible next type predicted
        err_time = 0  # i.e. goes back or stay in time (does not go forward)
        err_note = 0  # i.e. duplicated
        previous_type = tokens[0].split("_")[0]
        current_pos = -1
        current_pitches = []
        note_tokens_types = ['Pitch', 'NoteOn']

        # Init first note and current pitches if needed
        if previous_type in note_tokens_types:
            if previous_type in ['Pitch', 'NoteOn']:
                pitch_val = int(tokens[0].split("_")[1])
            else:  # PitchVel or PitchVelDur
                pitch_val = int(tokens[0].split("_")[1].split('-')[0])
            current_pitches.append(pitch_val)
        elif previous_type == 'Position':
            current_pos = int(tokens[0].split("_")[1])

        for token in tokens[1:]:
            event_type, event_value = token.split("_")[0], token.split("_")[1]

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type == 'Bar':  # reset
                    current_pos = -1
                    current_pitches = []
                elif event_type in ['TimeShift', 'Time-Shift', 'Rest']:
                    current_pitches = []
                elif event_type in note_tokens_types:
                    pitch_val = int(event_value)
                    if pitch_val in current_pitches:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches.append(pitch_val)
                elif event_type == 'Position':
                    if int(event_value) <= current_pos and previous_type != 'Rest':
                        err_time += 1  # token position value <= to the current position
                    else:
                        current_pos = int(event_value)
                        current_pitches = []
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
        Use kwargs to save any additional information within the JSON file.

        :param tokens: tokens, as list, numpy array, torch or tensorflow Tensor.
        :param path: path of the file to save.
        :param programs: (optional), programs of the associated tokens, should be
                        given as a tuples (int, bool) for (program, is_drum).
        :param kwargs: any additional information to save within the JSON file.
        """
        ids = []

        if isinstance(tokens, TokSequence):
            self.complete_sequence(tokens)
            ids = tokens.ids
        elif isinstance(tokens[0], TokSequence):
            for seq in tokens:
                self.complete_sequence(seq)
                ids.append(seq.ids)
        else:
            ids = convert_ids_tensors_to_list(ids)

        with open(path, "w") as outfile:
            json.dump(
                {
                    "ids": ids,
                    "programs": programs if programs is not None else [],
                    **kwargs,
                },
                outfile,
            )

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
        if (
            self.has_bpe and "_vocab_base" not in additional_attributes
        ):  # saves whole vocab if BPE
            additional_attributes["_vocab_base"] = self._vocab_base
            additional_attributes["_vocab_base_byte_to_token"] = self._vocab_base_byte_to_token
            additional_attributes["vocab_bpe"] = self._bpe_model.get_vocab()

        params = {
            "_pitch_range": (self._pitch_range.start, self._pitch_range.stop),
            "_beat_res": {f"{k1}_{k2}": v for (k1, k2), v in self._beat_res.items()},
            "_nb_velocities": len(self._velocities),
            "additional_tokens": self.additional_tokens,
            "special_tokens": self.special_tokens,
            "unique_track": self.unique_track,
            "has_bpe": self.has_bpe,
            "tokenization": self.__class__.__name__,
            "miditok_version": CURRENT_VERSION_PACKAGE,
            **additional_attributes,
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as outfile:
            json.dump(params, outfile, indent=4)

    def load_params(self, config_file_path: Union[str, Path]):
        r"""Loads the parameters of the tokenizer from a config file.

        :param config_file_path: path to the tokenizer config file (encoded as json).
        """
        with open(config_file_path) as param_file:
            params = json.load(param_file)

        params["_pitch_range"] = range(*params["_pitch_range"])

        for key, value in params.items():
            if key in ["tokenization", "miditok_version"]:
                continue
            elif key == "_beat_res":
                value = {
                    tuple(map(int, beat_range.split("_"))): res
                    for beat_range, res in value.items()
                }
            elif key == "additional_tokens":
                value["TimeSignature"] = value.get("TimeSignature", False)
            elif key == "_vocab_base":
                self._vocab_base = value
                self.__vocab_base_inv = {v: k for k, v in value.items()}
                continue
            elif key == "vocab_bpe":
                self._bpe_model = TokenizerFast(
                    BPE(
                        vocab=value,
                        merges=[],
                        dropout=None,
                        continuing_subword_prefix="",
                        end_of_word_suffix="",
                        fuse_unk=False,
                    )
                )
                continue
            elif key == "_vocab_base_byte_to_token":
                token_to_byte = {v: k for k, v in value.items()}
                self._vocab_base_id_to_byte = {i: token_to_byte[tok] for tok, i in self._vocab_base.items()}

            setattr(self, key, value)

    @property
    def is_multi_voc(self) -> bool:
        """Returns a bool indicating if the tokenizer uses embedding
        pooling, and so have multiple vocabularies.

        :return: True is the tokenizer uses embedding pooling else False
        """
        return isinstance(self._vocab_base, list)

    def __call__(self, obj: Any, *args, **kwargs):
        r"""Automatically tokenize a MIDI file, or detokenize a sequence of tokens.
        This will call the :py:func:`miditok.MIDITokenizer.midi_to_tokens` if you provide
        a MIDI object, or the :py:func:`miditok.MIDITokenizer.tokens_to_midi` method else.

        :param obj: a MIDI object or sequence of tokens.
        :return: the converted object.
        """
        if isinstance(obj, MidiFile):
            return self.midi_to_tokens(obj, *args, **kwargs)
        else:
            return self.tokens_to_midi(obj, *args, **kwargs)

    def __len__(self) -> int:
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling / have multiple vocabularies, it will return the **sum** of their lengths.
        Use the :py:func:`miditok.MIDITokenizer.len` property (``tokenizer.len``) to have the list of lengths.

        :return: length of the vocabulary.
        """
        if self.is_multi_voc:
            return sum([len(v) for v in self.vocab])
        return len(self.vocab)

    @property
    def len(self) -> Union[int, List[int]]:
        r"""Returns the length of the vocabulary. If the tokenizer uses embedding
        pooling / have multiple vocabularies, it will return the **list** of their lengths.
        Use the :py:func:`miditok.MIDITokenizer.__len__` magic method (``len(tokenizer)``)
        to get the sum of the lengths.

        :return: length of the vocabulary.
        """
        return [len(v) for v in self.vocab] if self.is_multi_voc else len(self.vocab)

    def __repr__(self):
        return (
            f'{len(self.len)} tokens {"(multi-voc) " if self.is_multi_voc else ""}'
            f'{"with BPE" if self.has_bpe else "without BPE"}'
        )

    def __getitem__(self, item: Union[int, str, Tuple[int, Union[int, str]]]) -> Union[str, int]:
        r"""Convert a token (int) to an event (str), or vice-versa.

        :param item: a token (int) or an event (str). For embedding pooling, you must
                provide a tuple where the first element in the index of the vocabulary.
        :return: the converted object.
        """
        if isinstance(item, tuple) and self.is_multi_voc:
            return self.__get_from_voc(item[1], item[0])
        else:
            return self.__get_from_voc(item)

    def __get_from_voc(self, item: Union[int, str], vocab_id: int = None) -> Union[int, str]:
        r"""Get element from the vocabulary.
        The method handles both token (int) <--> event (str) ways.

        :param item: item to get / index.
        :param vocab_id: index of the vocabulary associated to the token, if applicable. (default: None)
        :return: the associated value.
        """
        if isinstance(item, str):
            voc = self.vocab[vocab_id] if self.is_multi_voc else self.vocab
        else:
            voc = self.__vocab_base_inv[vocab_id] if self.is_multi_voc else self.__vocab_base_inv
        return voc[item]

    def __eq__(self, other) -> bool:
        r"""Checks if two tokenizers are identical. This is done by comparing their vocabularies,
        as they are built depending on most of their attributes.

        :param other: tokenizer to compare.
        :return: True if the vocabulary(ies) are identical, False otherwise.
        """
        if isinstance(other, MIDITokenizer):
            return self._vocab_base == other._vocab_base and \
                self._bpe_model.get_vocab() == other._bpe_model.get_vocab() and \
                self._vocab_base_byte_to_token == other._vocab_base_byte_to_token
        return False
