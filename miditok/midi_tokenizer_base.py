"""MIDI encoding base class and methods
TODO Control change messages (sustain, modulation, pitch bend)
TODO time signature changes tokens

"""
from abc import ABC, abstractmethod
import math
from pathlib import Path
import json
from random import choices
from copy import deepcopy
from typing import List, Tuple, Dict, Union, Callable, Optional, Any

import numpy as np
from tqdm import tqdm
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from .vocabulary import Vocabulary, Event
from .utils import remove_duplicated_notes, get_midi_programs
from .data_augmentation import data_augmentation_dataset
from .constants import TIME_DIVISION, CURRENT_VERSION_PACKAGE


def convert_tokens_tensors_to_list(func: Callable):
    """Decorator to handle tensor objects, for methods receiving tokens.
    Tokens have to be the first argument of the method (second for non-static).

    :param func: method to decorate
    :return: decorated method
    """

    def wrapper(*args, **kwargs):
        # Get tokens
        tokens_arg_id = 0 if not isinstance(args[0], MIDITokenizer) else 1
        tokens = args[tokens_arg_id]

        # Convert tokens to list if necessary
        if not isinstance(tokens, list):
            if type(tokens).__name__ in ["Tensor", "EagerTensor"]:
                tokens = tokens.numpy()
            if not isinstance(tokens, np.ndarray):
                raise TypeError(
                    "The tokens must be given as a list of integers, np.ndarray, PyTorch or Tensorflow tensor"
                )

            args = list(args)
            args[tokens_arg_id] = tokens.astype(
                int
            ).tolist()  # np.ndarray --> List[int]

        return func(*args, **kwargs)

    return wrapper


class MIDITokenizer(ABC):
    r"""MIDI encoding base class, containing common parameters to all encodings
    and common methods.

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, rests, tempo, time signature...)
    :param pad: will include a PAD token, used when training a model with batch of sequences of
            unequal lengths, and usually at index 0 of the vocabulary. (default: True)
    :param sos_eos: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary.
            (default: False)
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param unique_track: set to True if the tokenizer works only with a unique track.
            Tokens will be saved as a single track. This applies to representations that natively handle
            multiple tracks such as Octuple, resulting in a single "stream" of tokens for all tracks.
            This attribute will be saved in config files of the tokenizer. (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """

    def __init__(
        self,
        pitch_range: range,
        beat_res: Dict[Tuple[int, int], int],
        nb_velocities: int,
        additional_tokens: Dict[str, Union[bool, int, Tuple[int, int]]],
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        unique_track: bool = False,
        params: Union[str, Path, Dict[str, Any]] = None,
    ):
        # Initialize params
        self.vocab = None
        self.has_bpe = False
        if params is None:
            self.pitch_range = pitch_range
            self.beat_res = beat_res
            self.additional_tokens = additional_tokens
            self.nb_velocities = nb_velocities
            self._pad = pad
            self._sos_eos = sos_eos
            self._mask = mask
            self.unique_track = unique_track
        else:
            self.load_params(params)

        # Init duration and velocity values
        self.durations = self.__create_durations_tuples()
        self.velocities = np.linspace(0, 127, self.nb_velocities + 1, dtype=np.intc)[
            1:
        ]  # remove velocity 0
        self._first_beat_res = list(self.beat_res.values())[0]
        for beat_range, res in self.beat_res.items():
            if 0 in beat_range:
                self._first_beat_res = res
                break

        # Tempos
        self.tempos = np.zeros(1)
        if self.additional_tokens["Tempo"]:
            self.tempos = np.linspace(
                *self.additional_tokens["tempo_range"],
                self.additional_tokens["nb_tempos"],
                dtype=np.intc,
            )

        # Rests
        self.rests = []
        if self.additional_tokens["Rest"]:
            assert (
                self.additional_tokens["rest_range"][0] // 4 <= self._first_beat_res
            ), "The minimum rest value must be equal or superior to the initial beat resolution"
            self.rests = self.__create_rests()

        # Time Signatures
        self.time_signatures = []
        if self.additional_tokens["TimeSignature"]:
            self.time_signatures = self.__create_time_signatures()

        # Vocabulary and token types graph
        if (
            self.vocab is None
        ):  # in case it was already loaded by an overridden load_params method, such as with BPE
            self.vocab = self._create_vocabulary()
        self.tokens_types_graph = self._create_token_types_graph()

        # BPE attributes
        self.bpe_successions = {}
        if self.has_bpe:  # loaded from config file
            self._add_bpe_to_tokens_type_graph()
            self.set_bpe_tokens_successions()

        # Keep in memory durations in ticks for seen time divisions so these values
        # are not calculated each time a MIDI is processed
        self.durations_ticks = {}

        # Holds the tempo changes, time signature, time division and key signature of a
        # MIDI (being parsed) so that methods processing tracks can access them
        self.current_midi_metadata = {}  # needs to be updated each time a MIDI is read

    def preprocess_midi(self, midi: MidiFile):
        r"""Will process a MIDI file so it can be used to train a model.
        Its notes attribute (times, pitches, velocities) will be quantized and sorted, duplicated
        notes removed, as well as tempos.
        NOTE: empty tracks (with no note) will be removed from the MIDI object

        :param midi: MIDI object to preprocess
        """
        t = 0
        while t < len(midi.instruments):
            self.quantize_notes(
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
            self.quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

        if len(midi.time_signature_changes) == 0:  # can sometimes happen
            midi.time_signature_changes.append(
                TimeSignature(4, 4, 0)
            )  # 4/4 by default in this case
        if self.additional_tokens["TimeSignature"]:
            self.quantize_time_signatures(
                midi.time_signature_changes, midi.ticks_per_beat
            )

    def quantize_notes(
        self, notes: List[Note], time_division: int, pitch_range: range = None
    ):
        r"""Quantize the notes items, i.e. their pitch, velocity, start and end values.
        It shifts the notes so they start at times that match the quantization (e.g. 16 samples per bar)
        Notes with pitches outside of self.pitch_range will simply be deleted.

        :param notes: notes to quantize
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param pitch_range: pitch range from within notes should be (default None -> self.pitch_range)
        """
        if pitch_range is None:
            pitch_range = self.pitch_range
        ticks_per_sample = int(time_division / max(self.beat_res.values()))
        i = 0
        while i < len(notes):
            if notes[i].pitch not in pitch_range:
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
                self.velocities[
                    int(np.argmin(np.abs(self.velocities - notes[i].velocity)))
                ]
            )
            i += 1

    def quantize_tempos(self, tempos: List[TempoChange], time_division: int):
        r"""Quantize the times and tempo values of tempo change events.
        Consecutive identical tempo changes will be removed.

        :param tempos: tempo changes to quantize
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        """
        ticks_per_sample = int(time_division / max(self.beat_res.values()))
        prev_tempo = -1
        i = 0
        while i < len(tempos):
            # Quantize tempo value
            tempos[i].tempo = self.tempos[
                np.argmin(np.abs(self.tempos - tempos[i].tempo))
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
    def quantize_time_signatures(time_sigs: List[TimeSignature], time_division: int):
        r"""Quantize the time signature changes, delayed to the next bar.
        See MIDI 1.0 Detailed specifications, pages 54 - 56, for more information on
        delayed time signature messages.

        :param time_sigs: time signature changes to quantize
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
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

    def midi_to_tokens(
        self, midi: MidiFile, *args, **kwargs
    ) -> List[List[Union[int, List[int]]]]:
        r"""Converts a MIDI file in a tokens representation.
        NOTE: if you override this method, be sure to keep the first lines in your method

        :param midi: the MIDI objet to convert
        :return: the token representation, i.e. tracks converted into sequences of tokens
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self.durations_ticks:
            self.durations_ticks[midi.ticks_per_beat] = np.array(
                [
                    (beat * res + pos) * midi.ticks_per_beat // res
                    for beat, pos, res in self.durations
                ]
            )

        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self.current_midi_metadata = {
            "time_division": midi.ticks_per_beat,
            "tempo_changes": midi.tempo_changes,
            "time_sig_changes": midi.time_signature_changes,
            "key_sig_changes": midi.key_signature_changes,
        }

        # **************** OVERRIDE FROM HERE, KEEP THE LINES ABOVE IN YOUR METHOD ****************

        # Convert each track to tokens
        tokens = [self.track_to_tokens(track) for track in midi.instruments]

        return tokens

    @abstractmethod
    def track_to_tokens(self, track: Instrument) -> List[Union[int, List[int]]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        raise NotImplementedError

    def events_to_tokens(self, events: List[Event]) -> List[int]:
        r"""Converts a list of Event objects into a list of tokens
        Will apply BPE if it has been learned.

        :param events: list of Events objects to convert
        :return: list of corresponding tokens
        """
        tokens = [self.vocab.event_to_token[str(event)] for event in events]
        if self.has_bpe:  # this might take some time
            tokens = self.apply_bpe(tokens)
        return tokens

    def tokens_to_events(
        self, tokens: List[Union[int, List[int]]]
    ) -> List[Union[Event, List[Event]]]:
        r"""Convert a sequence of tokens in their respective event objects.
        BPE tokens will be decoded.

        :param tokens: sequence of tokens to convert
        :return: the sequence of corresponding events
        """
        events = []
        if self.is_multi_voc:  # multiple vocabularies
            for multi_token in tokens:
                multi_event = []
                for i, token in enumerate(multi_token):
                    name, val = self.vocab[i].token_to_event[token].split("_")
                    multi_event.append(Event(name, val))
                events.append(multi_event)
        else:
            tokens_ = self.decompose_bpe(tokens) if self.has_bpe else tokens
            for token in tokens_:
                name, val = self.vocab.token_to_event[token].split("_")
                events.append(Event(name, val))
        return events

    @convert_tokens_tensors_to_list
    def tokens_to_midi(
        self,
        tokens: Union[List, np.ndarray, Any],
        programs: Optional[List[Tuple[int, bool]]] = None,
        output_path: Optional[str] = None,
        time_division: Optional[int] = TIME_DIVISION,
    ) -> MidiFile:
        r"""Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        NOTE: With Remi, MIDI-Like, CP Word or other encoding methods that process tracks
        independently, only the tempo changes of the first track in tokens will be used

        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param programs: programs of the tracks
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
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
        tokens: List[Union[int, List[int]]],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and the possible tempo changes
        """
        raise NotImplementedError

    def add_sos_eos_to_seq(self, seq: List[int]):
        r"""Adds Start Of Sequence (SOS) and End Of Sequence EOS tokens to a sequence of tokens:
        SOS at the beginning, EOS at the end.

        :param seq: sequence of tokens
        """
        seq.insert(0, self.vocab["SOS_None"])
        seq.append(self.vocab["EOS_None"])

    @abstractmethod
    def _create_vocabulary(
        self, *args, **kwargs
    ) -> Union[Vocabulary, List[Vocabulary]]:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training
        NOTE 2: SOS and EOS tokens should be set to -1 and -2 respectively.
                use Vocabulary.add_sos_eos_to_vocab to add them

        :return: the vocabulary object
        """
        raise NotImplementedError

    @abstractmethod
    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Creates a dictionary for the directions of the token types of the encoding
        See other classes (REMI, MIDILike ...) for examples of how to implement it."""
        raise NotImplementedError

    def _add_special_tokens_to_types_graph(self, dic: Dict[str, List[str]]):
        r"""Inplace adds special tokens (PAD, EOS, SOS, MASK) types to the token types graph dictionary.

        :param dic: token types graph to add PAD type
        """
        if self._pad:
            for value in dic.values():
                value.append("PAD")
            dic["PAD"] = ["PAD"]
        if self._sos_eos:
            dic["SOS"] = list(dic.keys())
            dic["EOS"] = []
            for value in dic.values():
                value.append("EOS")
        if self._mask:
            dic["MASK"] = list(dic.keys())
            for value in dic.values():
                value.append("MASK")

    def _add_bpe_to_tokens_type_graph(self):
        r"""Adds BPE to the tokens_types_graph.
        You must manually call this method after loading a BPE tokenizer from params (config file) if
        you intend to use tokens_types_graph.
        """
        for val in self.tokens_types_graph.values():
            val.append("BPE")
        self.tokens_types_graph["BPE"] = list(self.tokens_types_graph.keys())

    def __create_durations_tuples(self) -> List[Tuple]:
        r"""Creates the possible durations in beat / position units, as tuple of the form:
        (beat, pos, res) where beat is the number of beats, pos the number of "samples"
        ans res the beat resolution considered (samples per beat)
        Example: (2, 5, 8) means the duration is 2 beat long + position 5 / 8 of the ongoing beat
        In pure ticks we have: duration = (beat * res + pos) * time_division // res
            Is equivalent to: duration = nb_of_samples * ticks_per_sample
        So in the last example, if time_division is 384: duration = (2 * 8 + 5) * 384 // 8 = 1008 ticks

        :return: the duration bins
        """
        durations = []
        for beat_range, beat_res in self.beat_res.items():
            durations += [
                (beat, pos, beat_res)
                for beat in range(*beat_range)
                for pos in range(beat_res)
            ]
        durations += [
            (max(max(self.beat_res)), 0, self.beat_res[max(self.beat_res)])
        ]  # the last one
        del durations[0]  # removes duration of 0
        return durations

    @staticmethod
    def _token_duration_to_ticks(token_duration: str, time_division: int) -> int:
        r"""Converts a duration token value of the form x.x.x, for beat.position.resolution,
        in ticks.
        Is also used for TimeShift tokens.

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

        :return: the rests
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

        :return: the time signatures
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

        Example: (10, 4), max_beat_res of 8, and nb_notes of 2 will convert the signature into (5, 4)

        :param numerator: time signature's numerator (bar length in beats)
        :param denominator: time signature's denominator (beat resolution)
        :return: the numerator and denominator of a reduced and decomposed time signature
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

        :param token_time_sig: TimeSig token value
        :return: the numerator and denominator of a time signature
        """
        numerator, denominator = map(int, token_time_sig.split("/"))
        return numerator, denominator

    def learn_bpe(
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

        :param tokens_path: path to token files to learn the BPE combinations from
        :param vocab_size: the new vocabulary size
        :param out_dir: directory to save the tokenizer's parameters and vocabulary after BPE learning is finished
        :param files_lim: limit of token files to use (default: None)
        :param save_converted_samples: will save in out_path the samples that have been used
                to create the BPE vocab. Files will keep the same name and relative path (default: True)
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
                [len(file["tokens"])]
                if self.unique_track
                else [len(track) for track in file["tokens"]]
            )

        def replace_token_in_seq(
            seq: List[int], succession: Tuple[int, int], new_event: str
        ):
            j = 0
            while j < len(seq) - 1:
                if tuple(seq[j: j + 2]) == succession:
                    seq[j] = self.vocab[f"BPE_{new_event}"]
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
                tracks = [sample["tokens"]] if self.unique_track else sample["tokens"]
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
                if self.vocab.token_to_event[token].split("_")[0] == "BPE":
                    prime_tokens_eq += map(
                        int,
                        self.vocab.token_to_event[token]
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
            self.vocab.add_event(
                Event(type_="BPE", time=0, value=final_event_val, desc="")
            )

            # Replace newly created token in learning samples
            for sample in samples:
                if self.unique_track:
                    replace_token_in_seq(
                        sample["tokens"], most_rec_tok_succession, final_event_val
                    )
                else:
                    for track in sample["tokens"]:
                        replace_token_in_seq(
                            track, most_rec_tok_succession, final_event_val
                        )

            # Compute metrics
            avg = []
            for sample in samples:
                if self.unique_track:
                    avg.append(len(sample["tokens"]))
                else:
                    avg += [len(track) for track in sample["tokens"]]
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
        self.set_bpe_tokens_successions()
        self._add_bpe_to_tokens_type_graph()
        self.vocab.update_token_types_indexes()
        if save_converted_samples:
            for sample, path in zip(samples, samples_paths):
                self.save_tokens(
                    sample["tokens"],
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

    def set_bpe_tokens_successions(self):
        """Creates the bpe_successions attributes, as a dictionary of the form {bpe_token: (tok1, tok2, tok3...)}"""
        self.bpe_successions = {
            tok: list(
                map(
                    int,
                    self.vocab.token_to_event[tok]
                    .split("_")[1]
                    .split(".")[0]
                    .split("-"),
                )
            )
            for tok in self.vocab.tokens_of_type("BPE")
        }

    def apply_bpe(self, tokens: List[int]) -> List[int]:
        r"""Converts a sequence of tokens into tokens with BPE.

        :param tokens: tokens to convert.
        :return:
        """
        if not self.has_bpe:
            return tokens

        previous_len = (
            len(tokens) + 1
        )  # + 1 to fool when entering the loop the first time
        while previous_len != len(
            tokens
        ):  # if this is True, it means no more BPE combinations is possible
            previous_len = len(
                tokens
            )  # length of the token sequence before applying BPE
            for (
                tok,
                token_succession,
            ) in (
                self.bpe_successions.items()
            ):  # loops over BPE tokens from the vocabulary
                occurrences = self.__find_subseq(tokens, token_succession)
                for idx in reversed(occurrences):
                    tokens[idx] = tok
                    for _ in range(len(token_succession) - 1):
                        del tokens[idx + 1]
        return tokens

    @staticmethod
    def __find_subseq(in_list: List[int], pattern: List[int]) -> List[int]:
        """Finds the locations of a pattern within a list.
        Adapted from: https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
        Related: https://www.reddit.com/r/learnpython/comments/2xqlwj/using_npwhere_to_find_subarrays/
        After testing, the numba jit version does not seem to be much faster.
        The conversion of python lists to numba.typed.List() seems to also take time.

        :param in_list: input list to analyze
        :param pattern: pattern to detect
        :return: indices of in_list where the pattern has been found
        """
        matches = []
        for i in range(len(in_list)):
            if in_list[i] == pattern[0] and in_list[i: i + len(pattern)] == pattern:
                matches.append(i)
        return matches

    def apply_bpe_to_dataset(
        self, dataset_path: Union[Path, str], out_path: Union[Path, str]
    ):
        r"""Apply BPE to an already tokenized dataset (with no BPE).

        :param dataset_path: path to token files to load
        :param out_path: output directory to save
        """
        if not self.has_bpe:
            return

        files_paths = list(Path(dataset_path).glob("**/*.json"))
        for path in tqdm(files_paths, desc="Applying BPE to dataset"):
            sample = self.load_tokens(path)
            sample_bpe = (
                self.apply_bpe(sample["tokens"])
                if self.unique_track
                else [self.apply_bpe(track) for track in sample["tokens"]]
            )
            self.save_tokens(
                sample_bpe,
                Path(out_path) / path.relative_to(dataset_path),
                sample["programs"],
            )

    def decompose_bpe(self, tokens: List[int]) -> List[int]:
        r"""Decomposes a sequence of tokens containing BP encoded tokens into "prime" tokens.
        It is an inplace operation.

        :param tokens: token sequence to decompose
        :return: decomposed token sequence
        """
        tokens = deepcopy(tokens)
        i = 0
        while i < len(tokens):
            token_type, token_val = self.vocab[tokens[i]].split("_")
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

        :param midi_paths: paths of the MIDI files
        :param out_dir: output directory to save the converted files
        :param validation_fn: a function checking if the MIDI is valid on your requirements
                            (e.g. time signature, minimum/maximum length, instruments ...)
        :param data_augment_offsets: data augmentation arguments, to be passed to the
            miditok.data_augmentation.data_augmentation_dataset method. Has to be given as a list / tuple
            of offsets pitch octaves, velocities, durations, and finaly their directions (up/down). (default: None)
        :param save_programs: will also save the programs of the tracks of the MIDI(default: True)
        :param logging: logs progress bar
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
            except Exception:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
                continue

            # Checks the time division is valid
            if midi.ticks_per_beat < max(self.beat_res.values()) * 4:
                continue
            # Passing the MIDI to validation tests if given
            if validation_fn is not None:
                if not validation_fn(midi):
                    continue

            # Converting the MIDI to tokens and saving them as json
            tokens = self.midi_to_tokens(midi)
            self.save_tokens(
                tokens,
                Path(out_dir, f"{Path(midi_path).stem}.json").with_suffix(".json"),
                get_midi_programs(midi) if save_programs else None,
            )

        # Perform data augmentation
        if data_augment_offsets is not None:
            data_augmentation_dataset(out_dir, self, *data_augment_offsets)

    def token_types_errors(
        self, tokens: List[int], consider_pad: bool = False
    ) -> float:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The implementation in MIDITokenizer class only checks the token types,
        in child class the methods also consider the position and pitch values.
        Overridden methods must call decompose_bpe at the beginning if BPE is used!

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        nb_tok_predicted = len(tokens)  # used to norm the score
        tokens = self.decompose_bpe(tokens) if self.has_bpe else tokens

        # Override from here

        err = 0
        previous_type = self.vocab.token_type(tokens[0])
        if consider_pad:
            for token in tokens[1:]:
                if (
                    self.vocab.token_type(token)
                    not in self.tokens_types_graph[previous_type]
                ):
                    err += 1
                previous_type = self.vocab.token_type(token)
        else:
            for token in tokens[1:]:
                if previous_type == "PAD":  # stop iteration at the first PAD token
                    break
                if (
                    self.vocab.token_type(token)
                    not in self.tokens_types_graph[previous_type]
                ):
                    err += 1
                previous_type = self.vocab.token_type(token)
        return err / nb_tok_predicted

    @staticmethod
    @convert_tokens_tensors_to_list
    def save_tokens(
        tokens: Union[List, np.ndarray, Any],
        path: Union[str, Path],
        programs: List[Tuple[int, bool]] = None,
        **kwargs,
    ):
        r"""Saves tokens as a JSON file.
        Use kwargs to save any additional information within the JSON file.

        :param tokens: tokens, as any format
        :param path: path of the file to save
        :param programs: (optional), programs of the associated tokens, should be
                        given as a tuples (int, bool) for (program, is_drum)
        :param kwargs: any additional information to save within the JSON file.
        """
        with open(path, "w") as outfile:
            json.dump(
                {
                    "tokens": tokens,
                    "programs": programs if programs is not None else [],
                    **kwargs,
                },
                outfile,
            )

    @staticmethod
    def load_tokens(path: Union[str, Path]) -> Union[List[Any], Dict]:
        r"""Loads tokens saved as JSON files.

        :param path: path of the file to load
        :return: the tokens, with the associated programs if saved with
        """
        with open(path) as file:
            return json.load(file)

    def save_params(
        self, out_path: Union[str, Path], additional_attributes: Dict = None
    ):
        r"""Saves the config / base parameters of the tokenizer in a file.
        Useful to keep track of how a dataset has been tokenized / encoded
        It will also save the name of the class used, i.e. the encoding strategy.
        NOTE: if you override this method, you should probably call it (super()) at the end
            and use the additional_attributes argument.
        NOTE 2: as json cant save tuples as keys, the beat ranges are saved as strings
        with the form startingBeat_endingBeat (underscore separating these two values)

        :param out_path: output path to save the file
        :param additional_attributes: any additional information to store in the config file.
                It can be used to override the default attributes saved in the parent method. (default: None)
        """
        if additional_attributes is None:
            additional_attributes = {}
        if (
            self.has_bpe and "vocab" not in additional_attributes
        ):  # saves whole vocab if BPE
            additional_attributes["vocab"] = self.vocab.token_to_event

        params = {
            "pitch_range": (self.pitch_range.start, self.pitch_range.stop),
            "beat_res": {f"{k1}_{k2}": v for (k1, k2), v in self.beat_res.items()},
            "nb_velocities": len(self.velocities),
            "additional_tokens": self.additional_tokens,
            "_pad": self._pad,
            "_sos_eos": self._sos_eos,
            "_mask": self._mask,
            "unique_track": self.unique_track,
            "tokenization": self.__class__.__name__,
            "miditok_version": CURRENT_VERSION_PACKAGE,
            **additional_attributes,
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as outfile:
            json.dump(params, outfile, indent=4)

    def load_params(self, params: Union[str, Path]):
        r"""Load parameters and set the encoder attributes

        :param params: can be a path to the parameter (json encoded) file
        """
        with open(params) as param_file:
            params = json.load(param_file)

        params["pitch_range"] = range(*params["pitch_range"])

        for key, value in params.items():
            if key in ["tokenization", "miditok_version"]:
                continue
            elif key == "beat_res":
                value = {
                    tuple(map(int, beat_range.split("_"))): res
                    for beat_range, res in value.items()
                }
            elif key == "additional_tokens":
                value["TimeSignature"] = value.get("TimeSignature", False)
            elif key == "vocab":
                self.vocab = Vocabulary(pad=False)
                for token, event in value.items():
                    if (
                        "-" in event and event.split("_")[0] != "BPE"
                    ):  # for config files created with < v1.3.3
                        event = "".join(
                            event.split("-")
                        )  # we remove hyphens to comply with the camelcase convention
                    self.vocab._token_to_event[
                        int(token)
                    ] = event  # we modify protected attribute to keep the exact
                    self.vocab._event_to_token[event] = int(
                        token
                    )  # same token <--> event pairs
                self.vocab.update_token_types_indexes()
                self.has_bpe = len(self.vocab.tokens_of_type("BPE")) > 0
                continue
            setattr(self, key, value)

        # when loading from params of miditok of previous versions
        if "_pad" not in params:  # miditok < v1.3.0
            self._pad = False
        if "_sos_eos" not in params:  # miditok < v1.2.0
            self._sos_eos = False
        if "_mask" not in params:  # miditok < v1.2.0
            self._mask = False

    @property
    def is_multi_voc(self) -> bool:
        return isinstance(self.vocab, list)

    def __call__(self, obj: Any, *args, **kwargs):
        if isinstance(obj, MidiFile):
            return self.midi_to_tokens(obj, *args, **kwargs)
        else:
            return self.tokens_to_midi(obj, *args, **kwargs)

    def __len__(self) -> int:
        if self.is_multi_voc:
            """warn('You are using a multi vocab tokenizer, returning the sum of the lengths of all vocabs.'
            'If you want the len per vocab, use the tokenizer.len property.')"""
            return sum([len(v) for v in self.vocab])
        return len(self.vocab)

    @property
    def len(self) -> Union[int, List[int]]:
        return [len(v) for v in self.vocab] if self.is_multi_voc else len(self.vocab)

    def __repr__(self):
        return (
            f'{len(self.len)} tokens {"(multi-voc)" if self.is_multi_voc else ""} '
            f'of {len(self.vocab.token_types)} types{" with BPE" if self.has_bpe else ""}'
        )

    def __getitem__(self, item: Union[int, str, Tuple[int, int]]) -> Union[str, int]:
        if isinstance(item, str):
            return self.vocab.event_to_token[item]
        elif isinstance(item, int):
            return self.vocab.token_to_event[item]
        elif isinstance(item, tuple) and self.is_multi_voc:
            return self.vocab[item[0]].token_to_event[item[1]]
        else:
            raise IndexError("The index must be an integer or a string")

    def __eq__(self, other) -> bool:
        """Checks if two tokenizers are identical. This is essentially done by comparing their vocabularies,
        as they are built depending on most of their attributes.

        :param other: tokenizer to compare.
        :return: True if the vocabulary(ies) are identical, False otherwise.
        """
        if isinstance(other, MIDITokenizer):
            return self.vocab == other.vocab
        return False
