from typing import List, Literal, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature

from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..classes import TokSequence, Event
from ..utils import detect_chords
from ..constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    SPECIAL_TOKENS,
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
    CHORD_MAPS,
)


class REMIPlus(MIDITokenizer):
    r"""REMI+ is extended REMI representation for general 
    multi-track, multisignature symbolic music sequences, introduced in 
    `FIGARO (Huang and Yang, 2020) <https://dl.acm.org/doi/10.1145/3394171.3413671>`,
    is a tokenization that represents notes as successions of *Pitch*, *Velocity* and *Duration*
    tokens, and time with *Bar* and *Position* tokens. A *Bar* token indicate that a new bar
    is beginning, and *Position* the current position within the current bar. The number of
    positions is determined by the ``beat_res`` argument, the maximum value will be used as
    resolution.
    **NOTE:** in the original paper, the tempo information is represented as the succession
    of two token types: a *TempoClass* indicating if the tempo is fast or slow, and a
    *TempoValue* indicating its value. MidiTok only uses one *Tempo* token for its value
    (see :ref:`Additional tokens`).

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
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS,
        params: Optional[Union[str, Path]] = None,
    ):
        self.encoder = []
        additional_tokens["Program"] = True  # required
        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            special_tokens,
            params=params,
        )

    def __notes_to_events(
        self, 
        notes_with_program: List[Tuple[Note, Tuple[int, bool]]]
    ) -> List[Event]:
        """Convert multi track data into one Token sequence
 
        args
        - notes_with_program: List[Tuple[Note, Tuple[int, bool]]]
            This contains track information in the Tuple[{program_number}, {is_drum}]
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(
            self.beat_res.values()
        )
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
                            value=str(current_bar + i + 1),
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
                                    time_sig_change.numerator, time_sig_change.denominator
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
                            time=note.start)
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

                # Adds chord events if specified
                if self.additional_tokens["Chord"] and not is_drum:
                    # Position before the Chord token
                    pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                    events.append(
                        Event(
                            type="Position",
                            value=pos_index,
                            time=note.start,
                            desc="ChordPosition",
                        )
                    )
                    events += detect_chords(
                        [n[0] for n  in notes_with_program],  # NOTE: very slow
                        self._current_midi_metadata["time_division"],
                        self._first_beat_res,
                    )

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
                    desc=note.pitch
                )
            )
            events.append(
                Event(
                    type="Pitch",
                    value=note.pitch,
                    time=note.start,
                    desc=note.pitch
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

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        events = self.__notes_to_events([
            (n, (track.program, track.is_drum)) for n in track.notes
        ])
        return TokSequence(events=events)  # type: ignore

    @_in_as_seq()
    def tokens_to_track(
        self,
        token_sequence: TokSequence,
        time_division: int = TIME_DIVISION,
    ) -> Tuple[List[Instrument], List[TempoChange], List[TimeSignature]]:
        r"""Converts a sequence of tokens into track objects

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        assert (
            time_division % max(self.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        tokens = token_sequence.tokens
        ticks_per_sample = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        # name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instruments: Dict[int, Instrument] = {} 
        # Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below
        time_signature_changes = [
            TimeSignature(4, 4, -1)
        ]  # mock the first time signature change to optimize below

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
                # If your encoding include tempo tokens, each Position token should be followed by
                # a tempo token, but if it is not the case this method will skip this step
                num, den = map(int, token.split("_")[1].split("/"))
                current_time_signature = time_signature_changes[-1]
                if num != current_time_signature.numerator and \
                        den != current_time_signature.denominator:
                    time_signature_changes.append(
                        TimeSignature(num, den, current_tick))
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
                                name="Drums" if program == -1 else MIDI_INSTRUMENTS[program]["name"]
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
        return list(instruments.values()), tempo_changes, time_signature_changes

    def _midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> TokSequence:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        Tokenization treating all tracks as a single token sequence might
        override this method, e.g. Octuple or PopMAG.

        :param midi: the MIDI objet to convert.
        :return: sequences of tokens.
        """
        # Convert each track to tokens
        notes_with_program = [
            ((note, (track.program, track.is_drum)))
            for track in midi.instruments for note in track.notes
        ]
        notes_with_program.sort(key=lambda n: (n[0].start, n[0].pitch))
        events = self.__notes_to_events(notes_with_program)
        tok_sequence = TokSequence(events=events)
        self.complete_sequence(tok_sequence)
        return tok_sequence

    def tokens_to_midi(
        self,
        tokens: TokSequence,
        output_path: Optional[str] = None,
        time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts multiple sequences of tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
                a Tensor (PyTorch and Tensorflow are supported), a numpy array or a Python list of ints.
                The first dimension represents tracks, unless the tokenizer handle tracks altogether as a
                single token sequence (e.g. Octuple, MuMIDI): tokenizer.unique_track == True.
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (miditoolkit.MidiFile).
        """
        midi = MidiFile(ticks_per_beat=time_division)
        tracks, tempo_changes, time_signature_changes = \
            self.tokens_to_track(tokens, time_division)
        midi.instruments = tracks
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


    def _create_base_vocabulary(self, sos_eos_tokens: bool = None) -> List[str]:
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
        vocab += [f"Bar_{i}" for i in range(0, 256)]  # NOTE: maximum bar length is 256

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
            vocab += [
                f"Chord_{i}" for i in range(3, 6)
            ]  # non recognized chords (between 3 and 5 notes only)
            vocab += [f"Chord_{chord_quality}" for chord_quality in CHORD_MAPS]

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
