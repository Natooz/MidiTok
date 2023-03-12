from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast
from itertools import product

import numpy as np
from miditoolkit import (
    Instrument,
    MidiFile,
    Note,
    TempoChange,
    TimeSignature,
    pianoroll,
)
from miditoolkit.pianoroll.parser import notes2pianoroll
from miditoolkit.pianoroll.utils import tochroma

from ..classes import Event, TokSequence
from ..constants import (
    ADDITIONAL_TOKENS,
    BEAT_RES,
    MIDI_INSTRUMENTS,
    NB_VELOCITIES,
    PITCH_RANGE,
    SPECIAL_TOKENS,
    TEMPO,
    TIME_DIVISION,
    TIME_SIGNATURE,
)
from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..utils import detect_chords


_PITCH_CLASSES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]
# define chord maps (required)
_CHORD_MAPS = {
    "maj": [0, 4],
    "min": [0, 3],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "dom": [0, 4, 7, 10],
}
# define chord insiders (+1)
_CHORD_INSIDERS = {"maj": [7], "min": [7], "dim": [9], "aug": [], "dom": []}
# define chord outsiders (-1)
_CHORD_OUTSIDERS_1 = {
    "maj": [2, 5, 9],
    "min": [2, 5, 8],
    "dim": [2, 5, 10],
    "aug": [2, 5, 9],
    "dom": [2, 5, 9],
}
# define chord outsiders (-2)
_CHORD_OUTSIDERS_2 = {
    "maj": [1, 3, 6, 8, 10],
    "min": [1, 4, 6, 9, 11],
    "dim": [1, 4, 7, 8, 11],
    "aug": [1, 3, 6, 7, 10],
    "dom": [1, 3, 6, 8, 11],
}


class REMIPlusChord:
    """
    Originally implemented in the REMI original repository
    <https://github.com/YatingMusic/remi/blob/master/chord_recognition.py>
    """

    @classmethod
    def __get_candidates(cls, chroma: np.ndarray) -> Dict[int, List[int]]:
        candidates: Dict[int, List[int]] = {}
        for index in range(len(chroma)):
            if chroma[index]:
                root_note = index
                _chroma = np.roll(chroma, -root_note)
                sequence = np.where(_chroma == 1)[0]
                candidates[root_note] = list(sequence)
        return candidates

    @classmethod
    def __get_score(
        cls, candidates: Dict[int, List[int]]
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        scores: Dict[int, int] = {}
        qualities: Dict[int, str] = {}
        for root_note, sequence in candidates.items():
            if 3 not in sequence and 4 not in sequence:
                scores[root_note] = -100
                qualities[root_note] = "None"
            elif 3 in sequence and 4 in sequence:
                scores[root_note] = -100
                qualities[root_note] = "None"
            else:
                # decide quality
                if 3 in sequence:
                    if 6 in sequence:
                        quality = "dim"
                    else:
                        quality = "min"
                elif 4 in sequence:
                    if 8 in sequence:
                        quality = "aug"
                    else:
                        if 7 in sequence and 10 in sequence:
                            quality = "dom"
                        else:
                            quality = "maj"
                else:
                    quality = ""
                # decide score rules
                maps = _CHORD_MAPS.get(quality, [])
                score = 0
                _notes = [n for n in sequence if n not in maps]
                for n in _notes:
                    if n in _CHORD_OUTSIDERS_1.get(quality, []):
                        score -= 1
                    elif n in _CHORD_OUTSIDERS_2.get(quality, []):
                        score -= 2
                    elif n in _CHORD_INSIDERS.get(quality, []):
                        score += 1
                scores[root_note] = score
                qualities[root_note] = quality
        return scores, qualities

    @classmethod
    def __find_chord(cls, pianoroll: np.ndarray) -> Tuple[str, str, str, int]:
        chroma: np.ndarray = tochroma(pianoroll=pianoroll)
        chroma = np.sum(chroma, axis=0)
        chroma = np.array([1 if c else 0 for c in chroma])
        if np.sum(chroma) == 0:
            return "None", "None", "None", 0
        else:
            candidates = cls.__get_candidates(chroma=chroma)
            scores, qualities = cls.__get_score(candidates=candidates)
            # bass note
            sorted_notes = []
            for i, v in enumerate(np.sum(pianoroll, axis=0)):
                if v > 0:
                    sorted_notes.append(int(i % 12))
            bass_note = sorted_notes[0]
            # root note
            __root_note = []
            _max = max(scores.values())
            for _root_note, score in scores.items():
                if score == _max:
                    __root_note.append(_root_note)
            if len(__root_note) == 1:
                root_note = __root_note[0]
            else:
                for n in sorted_notes:
                    if n in __root_note:
                        root_note = n
                        break
                return "None", "None", "None", 0  # no root found
            # quality
            quality = qualities.get(root_note, "None")
            sequence = candidates.get(root_note, [])
            # score
            score = scores.get(root_note, 0)
            return (
                _PITCH_CLASSES[root_note],
                quality,
                _PITCH_CLASSES[bass_note],
                score,
            )

    @classmethod
    def __solve(
        cls,
        candidates: Dict[int, Dict[int, Tuple[int, float, int, float]]],
        max_tick: int,
    ) -> List[Tuple[int, int, str]]:
        chords: List[Tuple[int, int, str]] = []
        start_tick = 0
        while start_tick < max_tick:
            _candidates = candidates.get(start_tick, {})
            _candidates = sorted(_candidates.items(), key=lambda x: (x[1][-1], x[0]))
            # choose
            end_tick, (root_note, quality, bass_note, _) = _candidates[-1]
            if root_note == bass_note:
                chord = "{}:{}".format(root_note, quality)
            else:
                chord = "{}:{}/{}".format(root_note, quality, bass_note)
            chords.append((start_tick, end_tick, chord))
            start_tick = end_tick
        # remove :None
        __temp = copy(chords)
        while ":None" in str(__temp[0][-1]):
            try:
                _new_head = (__temp[0][0], __temp[1][1], __temp[1][2])
                del __temp[0]  # delete None
                __temp = [_new_head] + __temp[1:]
            except:
                return []
        __temp2 = []
        for chord in __temp:
            if ":None" not in str(chord[-1]):
                __temp2.append(chord)
            else:
                # __temp2[-1][1] = chord[1]
                __temp2 = __temp2[:-1] + [(__temp2[-1][0], chord[1], __temp2[-1][2])]
        return __temp2

    @classmethod
    def extract(cls, notes: List[Note]) -> List[Tuple[int, int, str]]:
        # read
        max_tick = max([n.end for n in notes])
        ticks_per_beat = 480
        pianoroll = notes2pianoroll(
            note_stream_ori=notes, max_tick=max_tick, ticks_per_beat=ticks_per_beat
        )
        pianoroll = cast(np.ndarray, pianoroll)
        # get lots of candidates
        candidates = {}
        # the shortest: 2 beat (1/2 bar in 4/4), longest: 4 beat (1bar in 4/4)
        for interval in [4, 2]:
            for start_tick in range(0, max_tick, ticks_per_beat):
                end_tick = int(ticks_per_beat * interval + start_tick)
                if end_tick > max_tick:
                    end_tick = max_tick
                part_pianoroll = pianoroll[start_tick:end_tick, :]
                # find chord
                root_note, quality, bass_note, score = cls.__find_chord(
                    pianoroll=part_pianoroll
                )
                # save
                if start_tick not in candidates:
                    candidates[start_tick] = {}
                    candidates[start_tick][end_tick] = (
                        root_note,
                        quality,
                        bass_note,
                        score,
                    )
                else:
                    if end_tick not in candidates[start_tick]:
                        candidates[start_tick][end_tick] = (
                            root_note,
                            quality,
                            bass_note,
                            score,
                        )
        chords = cls.__solve(candidates=candidates, max_tick=max_tick)
        return chords


class REMIPlus(MIDITokenizer):
    r"""REMI+ is extended REMI representation (Huang and Yang) for general
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`, which
    represents notes as successions of *Program* (originally *Instrument* in the paper),
    *Pitch*, *Velocity* and *Duration* tokens, and time with *Bar* and *Position* tokens.
    A *Bar* token indicate that a new bar is beginning, and *Position* the current
    position within the current bar. The number of positions is determined by
    the ``beat_res`` argument, the maximum value will be used as resolution.

    :param pitch_range: range of MIDI pitches to use
    :param beat_res: beat resolutions, as a dictionary:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys are tuples indicating a range of beats, ex 0 to 3 for the first bar, and
            the values are the resolution to apply to the ranges, in samples per beat, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: additional tokens (chords, time signature, rests, tempo...) to use,
            to be given as a dictionary. (default: None is used)
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "BOS", "EOS", "MASK"]``)
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    :param num_bars: Maximum number of bars ("Bar_0", "Bar_1",...,"Bar_{num_bars-1}").
            If None passed, creates "Bar_None" token only in vocabulary
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, bool | int | Tuple[int, int]] = ADDITIONAL_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS,
        params: Optional[Union[str, Path]] = None,
        num_bars: Optional[int] = None,
    ):
        self.encoder = []
        additional_tokens["Program"] = True  # required
        self.num_bars = num_bars
        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            special_tokens,
            unique_track=True,  # handles multi-track sequences in single stream
            params=params,  # type: ignore
        )

    def __notes_to_events(
        self, notes_with_program: List[Tuple[Note, Tuple[int, bool]]]
    ) -> List[Event]:
        """Convert multi track notes into one Token sequence.

        :param notes_with_program: List[Tuple[Note, program]]
                Note is a instance of `miditoolkit.Note`, and the program
                is a tuple of (program_number: int, is_drum: bool).
        :return: sequences of Event.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(self.beat_res.values())
        ticks_per_bar = self._current_midi_metadata["time_division"] * 4
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        min_rest = (
            self._current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.additional_tokens["Rest"]
            else 0
        )
        # Creates events
        events: List[Event] = []
        previous_tick = -1
        previous_note_end = (
            notes_with_program[0][0].start + 1
        )  # so that no rest is created before the first note
        current_bar = -1
        current_tempo_idx = 0
        current_tempo = self._current_midi_metadata["tempo_changes"][
            current_tempo_idx
        ].tempo
        current_time_sig_idx = 0
        current_time_sig_tick = 0
        current_time_sig_bar = 0
        time_sig_change = self._current_midi_metadata["time_sig_changes"][
            current_time_sig_idx
        ]
        current_time_sig = self._reduce_time_signature(
            time_sig_change.numerator, time_sig_change.denominator
        )
        # Run chord extraction for whole note sequences before tokenization
        current_chord_idx = 0
        current_chord = ""  # e.g. C#:min/A
        if self.additional_tokens.get("Chord", False):  # "Chord" in additional tokens
            chord_results = REMIPlusChord.extract([np[0] for np in notes_with_program])
        else:
            chord_results = None

        for note, (program_num, is_drum) in notes_with_program:
            if note.start != previous_tick:
                # (Rest)
                if (
                    self.additional_tokens["Rest"]
                    and note.start > previous_note_end
                    and note.start - previous_note_end >= min_rest
                ):
                    previous_tick = previous_note_end
                    rest_beat, rest_pos = divmod(
                        note.start - previous_tick,
                        self._current_midi_metadata["time_division"],
                    )
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)

                    if rest_beat > 0:
                        events.append(
                            Event(
                                type="Rest",
                                value=f"{rest_beat}.0",
                                time=previous_note_end,
                                desc=f"{rest_beat}.0",
                            )
                        )
                        previous_tick += (
                            rest_beat * self._current_midi_metadata["time_division"]
                        )

                    while rest_pos >= self.rests[0][1]:
                        rest_pos_temp = min(
                            [r[1] for r in self.rests], key=lambda x: abs(x - rest_pos)
                        )
                        events.append(
                            Event(
                                type="Rest",
                                value=f"0.{rest_pos_temp}",
                                time=previous_note_end,
                                desc=f"0.{rest_pos_temp}",
                            )
                        )
                        previous_tick += round(rest_pos_temp * ticks_per_sample)
                        rest_pos -= rest_pos_temp

                    current_bar = previous_tick // ticks_per_bar

                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    events.append(
                        Event(
                            type="Bar",
                            value=str(current_bar + i + 1) if self.num_bars else "None",
                            time=(current_bar + i + 1) * ticks_per_bar,
                            desc=0,
                        )
                    )
                current_bar += nb_new_bars

                # (TimeSignature)
                if self.additional_tokens["TimeSignature"]:
                    # If the current time signature is not the last one
                    if current_time_sig_idx + 1 < len(
                        self._current_midi_metadata["time_sig_changes"]
                    ):
                        # Will loop over incoming time signature changes
                        for time_sig_change in self._current_midi_metadata[
                            "time_sig_changes"
                        ][current_time_sig_idx + 1 :]:
                            # If this time signature change happened before the current moment
                            if time_sig_change.time <= note.start:
                                current_time_sig = self._reduce_time_signature(
                                    time_sig_change.numerator,
                                    time_sig_change.denominator,
                                )
                                current_time_sig_idx += 1  # update time signature value (might not change) and index
                                current_time_sig_bar += (
                                    time_sig_change.time - current_time_sig_tick
                                ) // ticks_per_bar
                                current_time_sig_tick = time_sig_change.time
                                ticks_per_bar = time_division * current_time_sig[0]
                            elif time_sig_change.time > note.start:
                                break  # this time signature change is beyond the current time step, we break the loop
                    if nb_new_bars > 0:  # put a TimeSig token after the Bar token
                        events.append(
                            Event(
                                type="TimeSig",
                                value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                                time=note.start,
                            )
                        )

                # (Tempo)
                if self.additional_tokens["Tempo"]:
                    # If the current tempo is not the last one
                    if current_tempo_idx + 1 < len(
                        self._current_midi_metadata["tempo_changes"]
                    ):
                        # Will loop over incoming tempo changes
                        for tempo_change in self._current_midi_metadata[
                            "tempo_changes"
                        ][current_tempo_idx + 1 :]:
                            # If this tempo change happened before the current moment
                            if tempo_change.time <= note.start:
                                current_tempo = tempo_change.tempo
                                current_tempo_idx += (
                                    1  # update tempo value (might not change) and index
                                )
                            else:  # <==> elif tempo_change.time > previous_tick:
                                break  # this tempo change is beyond the current time step, we break the loop
                    if nb_new_bars > 0:  # after the new Bar token
                        # Position before the Tempo token
                        pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                        events.append(
                            Event(
                                type="Position",
                                value=pos_index,
                                time=note.start,
                                desc="TempoPosition",
                            )
                        )
                        events.append(
                            Event(
                                type="Tempo",
                                value=current_tempo,
                                time=note.start,
                                desc=note.start,
                            )
                        )

                # (Chord)
                if chord_results is not None:
                    """
                    chord_results: list of chord with ticks range
                    [(start_tick, end_tick, chord_string)...]
                    """
                    pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                    if nb_new_bars > 0 and current_chord != "":
                        # put chord token in every bar start
                        events.append(
                            Event(
                                type="Position",
                                value=pos_index,
                                time=note.start,
                                desc="ChordPosition",
                            )
                        )
                        events.append(
                            Event(
                                type="Chord",
                                value=current_chord,
                                time=note.start,
                                desc=note.start,
                            )
                        )
                    elif (
                        note.start > chord_results[current_chord_idx][0]
                        and note.start < chord_results[current_chord_idx][1]
                    ):
                        current_chord = chord_results[current_chord_idx][-1]
                    elif note.start > chord_results[current_chord_idx][1]:
                        # chord changed within a bar
                        current_chord_idx = min(
                            current_chord_idx + 1, len(chord_results) - 1
                        )
                        if current_chord_idx <= len(chord_results):
                            current_chord = ""

                previous_tick = note.start

            # Position
            pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
            events.append(
                Event(
                    type="Position",
                    value=pos_index,
                    time=note.start,
                    desc="NotePosition",
                )
            )
            # Pitch / Velocity / Duration
            events.append(
                Event(
                    type="Program",
                    value=-1 if is_drum else program_num,
                    time=note.start,
                    desc=note.pitch,
                )
            )
            events.append(
                Event(type="Pitch", value=note.pitch, time=note.start, desc=note.pitch)
            )
            events.append(
                Event(
                    type="Velocity",
                    value=note.velocity,
                    time=note.start,
                    desc=f"{note.velocity}",
                )
            )
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
            previous_note_end = max(previous_note_end, note.end)

        events.sort(key=lambda x: (x.time, self._order(x)))
        return events

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).

        NOTE: REMI+ is REMI-based extended representation for multi-track encodings in single sequence. Then if you'd
        like to get only single-track tokens, use REMI.

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        events = self.__notes_to_events(
            [(n, (track.program, track.is_drum)) for n in track.notes]
        )
        return TokSequence(events=events)  # type: ignore

    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> None:
        r"""NOT RELEVANT / IMPLEMENTED FOR REMIPlus
        Use tokens_to_midi instead

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: None
        """
        pass

    def _midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> TokSequence:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.

        :param midi: the MIDI objet to convert.
        :return: sequences of tokens.
        """
        # Convert each track to tokens
        notes_with_program = [
            ((note, (track.program, track.is_drum)))
            for track in midi.instruments
            for note in track.notes
        ]
        notes_with_program.sort(key=lambda n: (n[0].start, n[0].pitch))
        events = self.__notes_to_events(notes_with_program)
        tok_sequence = TokSequence(events=cast(List[Union[Event, List[Event]]], events))
        self.complete_sequence(tok_sequence)
        return tok_sequence

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        _=None,
        output_path: Optional[str] = None,
        time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        tokens = cast(TokSequence, tokens)
        midi = MidiFile(ticks_per_beat=time_division)
        assert (
            time_division % max(self.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        tokens = cast(List[str], tokens.tokens)  # for reducing type errors
        ticks_per_sample = time_division // max(self.beat_res.values())

        # RESULTS
        instruments: Dict[int, Instrument] = {}
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below
        time_signature_changes = [
            TimeSignature(*TIME_SIGNATURE, 0)
        ]  # mock the first time signature change to optimize below
        ticks_per_bar = time_division * TIME_SIGNATURE[0]  # init

        current_tick = 0
        current_bar = -1
        previous_note_end = 0
        for ti, token in enumerate(tokens):
            if token.split("_")[0] == "Bar":
                current_bar += 1
                current_tick = current_bar * ticks_per_bar
            elif token.split("_")[0] == "Rest":
                beat, pos = map(int, tokens[ti].split("_")[1].split("."))
                if (
                    current_tick < previous_note_end
                ):  # if in case successive rest happen
                    current_tick = previous_note_end
                current_tick += beat * time_division + pos * ticks_per_sample
                current_bar = current_tick // ticks_per_bar
            elif token.split("_")[0] == "Position":
                if current_bar == -1:
                    current_bar = (
                        0  # as this Position token occurs before any Bar token
                    )
                current_tick = (
                    current_bar * ticks_per_bar
                    + int(token.split("_")[1]) * ticks_per_sample
                )
            elif token.split("_")[0] == "Tempo":
                # If your encoding include tempo tokens, each Position token should be followed by
                # a tempo token, but if it is not the case this method will skip this step
                tempo = int(token.split("_")[1])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            elif token.split("_")[0] == "TimeSig":
                num, den = self._parse_token_time_signature(token.split("_")[1])
                current_time_signature = time_signature_changes[-1]
                if (
                    num != current_time_signature.numerator
                    and den != current_time_signature.denominator
                ):
                    time_signature_changes.append(TimeSignature(num, den, current_tick))
            elif token.split("_")[0] == "Pitch":
                try:
                    if (
                        tokens[ti + 1].split("_")[0] == "Velocity"
                        and tokens[ti + 2].split("_")[0] == "Duration"
                        and tokens[ti - 1].split("_")[0] == "Program"
                    ):
                        program = int(tokens[ti - 1].split("_")[1])
                        pitch = int(tokens[ti].split("_")[1])
                        vel = int(tokens[ti + 1].split("_")[1])
                        duration = self._token_duration_to_ticks(
                            tokens[ti + 2].split("_")[1], time_division
                        )
                        if program not in instruments.keys():
                            instruments[program] = Instrument(
                                program=0 if program == -1 else program,
                                is_drum=program == -1,
                                name="Drums"
                                if program == -1
                                else MIDI_INSTRUMENTS[program]["name"],
                            )
                        instruments[program].notes.append(
                            Note(vel, pitch, current_tick, current_tick + duration)
                        )
                        previous_note_end = max(
                            previous_note_end, current_tick + duration
                        )
                except (
                    IndexError
                ):  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        if len(time_signature_changes) > 1:
            del time_signature_changes[0]  # delete mocked time signature change
        time_signature_changes[0].time = 0
        # create MidiFile
        midi.instruments = list(instruments.values())
        midi.tempo_changes = tempo_changes
        midi.time_signature_changes = time_signature_changes
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

    def _create_base_vocabulary(
        self, sos_eos_tokens: Optional[bool] = None
    ) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = []

        # BAR
        if self.num_bars:
            vocab += [f"Bar_{i}" for i in range(0, self.num_bars)]
        else:
            vocab += ["Bar_None"]

        # PITCH
        vocab += [f"Pitch_{i}" for i in self.pitch_range]

        # VELOCITY
        vocab += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab += [f"Position_{i}" for i in range(nb_positions)]

        # TIME SIGNATURE
        if self.additional_tokens["TimeSignature"]:
            vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]

        # CHORD
        if self.additional_tokens["Chord"]:
            # extract combination mapping in root and chords
            for root in _PITCH_CLASSES:
                for quality in _CHORD_MAPS.keys():
                    vocab.append(f"Chord_{root}:{quality}")
                    for base in _PITCH_CLASSES:
                        if base != root:
                            # add fraction chords
                            vocab.append(f"Chord_{root}:{quality}/{base}")

        # REST
        if self.additional_tokens["Rest"]:
            vocab += [f'Rest_{".".join(map(str, rest))}' for rest in self.rests]

        # TEMPO
        if self.additional_tokens["Tempo"]:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        # PROGRAM
        if self.additional_tokens["Program"]:
            vocab += [f"Program_{program}" for program in range(-1, 128)]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.

        :return: the token types transitions dictionary
        """
        dic: Dict[str, List[str]] = dict()

        dic["Bar"] = ["Position", "Bar"]
        dic["Position"] = ["Program"]
        dic["Program"] = ["Pitch"]
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Program", "Position", "Bar"]

        if self.additional_tokens["TimeSignature"]:
            dic["Bar"] = ["TimeSig", "Bar"]
            dic["TimeSig"] = ["Position"]

        if self.additional_tokens["Chord"]:
            dic["Chord"] = ["Position"]
            dic["Position"] += ["Chord"]

        if self.additional_tokens["Tempo"]:
            dic["Tempo"] = ["Position"]
            dic["Position"] += ["Tempo"]

        if self.additional_tokens["Rest"]:
            dic["Rest"] = ["Rest", "Position", "Bar"]
            dic["Duration"] += ["Rest"]

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Bar":
            return 0
        elif x.type == "TimeSig":
            return 1
        elif x.type == "Position" and x.desc == "TempoPosition":
            return 2
        elif x.type == "Tempo":
            return 3
        elif x.type == "Position" and x.desc == "ChordPosition":
            return 4
        elif x.type == "Chord":
            return 5
        elif x.type == "Rest":
            return 7
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 8
