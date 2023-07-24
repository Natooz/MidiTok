from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from ..midi_tokenizer import MIDITokenizer, _in_as_seq
from ..classes import TokSequence, Event
from ..utils import detect_chords
from ..constants import (
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
    TIME_SIGNATURE,
)


class TSD(MIDITokenizer):
    r"""TSD, for Time Shift Duration, is similar to MIDI-Like :ref:`MIDI-Like`
    but uses explicit *Duration* tokens to represent note durations, which have
    showed `better results than with NoteOff tokens <https://arxiv.org/abs/2002.00212>`_.
    **Note:** as TSD uses *TimeShifts* events to move the time from note to
    note, it could be unsuited for tracks with long pauses. In such case, the
    maximum *TimeShift* value will be used.
    """

    def _tweak_config_before_creating_voc(self):
        self.config.use_time_signatures = False
        if self.config.use_programs:
            self.one_token_stream = True

    def __notes_to_events(self, track: Instrument) -> List[Event]:
        r"""Converts notes of a track (``miditoolkit.Instrument``) into a sequence of `Event` objects.

        :param track: MIDI track to convert
        :return: sequence of corresponding Events
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        program = track.program if not track.is_drum else -1
        events = []

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

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Note On / Velocity / Duration
            if self.config.use_programs:
                events.append(
                    Event(type="Program", value=program, time=note.start, desc=note.end)
                )
            events.append(
                Event(type="Pitch", value=note.pitch, time=note.start, desc=note.end)
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

        return events

    def __add_time_note_events(self, events: List[Event]) -> List[Event]:
        r"""
        Takes a sequence of note events (containing optionally Chord, Tempo and TimeSignature tokens),
        and insert (not inplace) time tokens (TimeShift, Rest) to complete the sequence.

        :param events: note events to complete.
        :return: the same events, with time events inserted.
        """
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        ticks_per_sample = self._current_midi_metadata["time_division"] / max(
            self.config.beat_res.values()
        )
        min_rest = (
            self._current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.config.use_rests
            else 0
        )

        # Add time events
        all_events = events.copy()
        previous_tick = 0
        previous_note_end = events[0].time + 1
        for e, event in enumerate(events.copy()):
            # No time shift
            if event.time != previous_tick:
                # (Rest)
                if (
                    event.type in ["Pitch", "Tempo", "TimeSig"]
                    and self.config.use_rests
                    and event.time - previous_note_end >= min_rest
                ):
                    # untouched tick value to the order is not messed after sorting
                    # in case of tempo change, we need to take its time as reference
                    rest_tick = max(previous_note_end, previous_tick)
                    rest_beat, rest_pos = divmod(
                        event.time - rest_tick,
                        self._current_midi_metadata["time_division"],
                    )
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)
                    previous_tick = rest_tick

                    if rest_beat > 0:
                        all_events.append(
                            Event(
                                type="Rest",
                                value=f"{rest_beat}.0",
                                time=rest_tick,
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
                        all_events.append(
                            Event(
                                type="Rest",
                                value=f"0.{rest_pos_temp}",
                                time=rest_tick,
                                desc=f"0.{rest_pos_temp}",
                            )
                        )
                        previous_tick += round(rest_pos_temp * ticks_per_sample)
                        rest_pos -= rest_pos_temp

                    # Adds an additional time shift if needed
                    if rest_pos > 0:
                        time_shift = round(rest_pos * ticks_per_sample)
                        index = np.argmin(np.abs(dur_bins - time_shift))
                        all_events.append(
                            Event(
                                type="TimeShift",
                                value=".".join(map(str, self.durations[index])),
                                time=previous_tick,
                                desc=f"{time_shift} ticks",
                            )
                        )

                # Time shift
                else:
                    time_shift = event.time - previous_tick
                    index = np.argmin(np.abs(dur_bins - time_shift))
                    all_events.append(
                        Event(
                            type="TimeShift",
                            value=".".join(map(str, self.durations[index])),
                            time=previous_tick,
                            desc=f"{time_shift} ticks",
                        )
                    )
                previous_tick = event.time

            # Update max offset time of the notes encountered
            if event.type == "Pitch":
                previous_note_end = max(previous_note_end, event.desc)

        # Sort the tokens so that they come in the good order
        all_events.sort(key=lambda x: (x.time, self._order(x)))
        return all_events

    def _midi_to_tokens(
        self, midi: MidiFile, *args, **kwargs
    ) -> Union[TokSequence, List[TokSequence]]:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.

        :param midi: the MIDI objet to convert.
        :return: a :class:`miditok.TokSequence` if `tokenizer.one_token_stream` is true, else a list of
                :class:`miditok.TokSequence` objects.
        """
        # Convert each track to tokens
        all_events = []

        # Adds note tokens
        for track in midi.instruments:
            note_events = self.__notes_to_events(track)
            if self.one_token_stream:
                all_events += note_events
            else:
                all_events.append(note_events)
        # Adds tempo events if specified
        if self.config.use_tempos:
            tempo_events = []
            for tempo_change in self._current_midi_metadata["tempo_changes"]:
                tempo_events.append(
                    Event(
                        type="Tempo",
                        value=tempo_change.tempo,
                        time=tempo_change.time,
                        desc=tempo_change.tempo,
                    )
                )
            if self.one_token_stream:
                all_events += tempo_events
            else:
                for i in range(len(all_events)):
                    all_events[i] += tempo_events

        # Add time events
        if self.one_token_stream:
            all_events.sort(key=lambda x: x.time)
            all_events = self.__add_time_note_events(all_events)
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                all_events[i].sort(key=lambda x: x.time)
                all_events[i] = self.__add_time_note_events(all_events[i])
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

    def tokens_to_track(
        self,
        tokens: TokSequence,
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        pass

    def track_to_tokens(self, track: Instrument) -> TokSequence:
        pass

    @_in_as_seq()
    def tokens_to_midi(
        self,
        tokens: Union[
            Union[TokSequence, List, np.ndarray, Any],
            List[Union[TokSequence, List, np.ndarray, Any]],
        ],
        programs: Optional[List[Tuple[int, bool]]] = None,
        output_path: Optional[str] = None,
        time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param programs: programs of the tracks. If none is given, will default to piano, program 0. (default: None)
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.one_token_stream:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        midi = MidiFile(ticks_per_beat=time_division)
        assert (
            time_division % max(self.config.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.config.beat_res.values())}"
        ticks_per_sample = time_division // max(self.config.beat_res.values())

        # RESULTS
        instruments: Dict[int, Instrument] = {}
        tempo_changes = [TempoChange(TEMPO, -1)]
        time_signature_changes = [TimeSignature(*TIME_SIGNATURE, 0)]

        current_tick = 0
        current_program = 0
        previous_note_end = 0
        for si, seq in enumerate(tokens):
            # Set track / sequence program if needed
            if not self.one_token_stream:
                current_tick = 0
                previous_note_end = 0
                if programs is not None:
                    current_program = -1 if programs[si][1] else programs[si][0]

            # Decode tokens
            for ti, token in enumerate(seq):
                if token.split("_")[0] == "TimeShift":
                    current_tick += self._token_duration_to_ticks(
                        token.split("_")[1], time_division
                    )
                elif token.split("_")[0] == "Rest":
                    beat, pos = map(int, seq[ti].split("_")[1].split("."))
                    if (
                        current_tick < previous_note_end
                    ):  # if in case successive rest happen
                        current_tick = previous_note_end
                    current_tick += beat * time_division + pos * ticks_per_sample
                elif token.split("_")[0] == "Pitch":
                    try:
                        if (
                            seq[ti + 1].split("_")[0] == "Velocity"
                            and seq[ti + 2].split("_")[0] == "Duration"
                        ):
                            pitch = int(seq[ti].split("_")[1])
                            vel = int(seq[ti + 1].split("_")[1])
                            duration = self._token_duration_to_ticks(
                                seq[ti + 2].split("_")[1], time_division
                            )
                            if current_program not in instruments.keys():
                                instruments[current_program] = Instrument(
                                    program=0
                                    if current_program == -1
                                    else current_program,
                                    is_drum=current_program == -1,
                                    name="Drums"
                                    if current_program == -1
                                    else MIDI_INSTRUMENTS[current_program]["name"],
                                )
                            instruments[current_program].notes.append(
                                Note(vel, pitch, current_tick, current_tick + duration)
                            )
                            previous_note_end = max(
                                previous_note_end, current_tick + duration
                            )
                    except (
                        IndexError
                    ):  # A well constituted sequence should not raise an exception
                        pass  # However with generated sequences this can happen, or if the sequence isn't finished
                elif token.split("_")[0] == "Program":
                    current_program = int(token.split("_")[1])
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
                        time_signature_changes.append(
                            TimeSignature(num, den, current_tick)
                        )
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

    def _create_base_vocabulary(self, sos_eos_tokens: bool = False) -> List[str]:
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

        # NOTE ON
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # VELOCITY
        vocab += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # TIME SHIFTS
        vocab += [
            f'TimeShift_{".".join(map(str, self.durations[i]))}'
            for i in range(len(self.durations))
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

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Pitch", "TimeShift"]
        dic["TimeShift"] = ["Pitch"]

        if self.config.use_chords:
            dic["Chord"] = ["Pitch"]
            dic["TimeShift"] += ["Chord"]

        if self.config.use_tempos:
            dic["TimeShift"] += ["Tempo"]
            dic["Tempo"] = ["Pitch", "TimeShift"]
            if self.config.use_chords:
                dic["Tempo"] += ["Chord"]

        if self.config.use_rests:
            dic["Rest"] = ["Rest", "Pitch", "TimeShift"]
            dic["Duration"] += ["Rest"]
            if self.config.use_chords:
                dic["Rest"] += ["Chord"]
            if self.config.use_tempos:
                dic["Rest"] += ["Tempo"]
                dic["Tempo"] += ["Rest"]
            if self.config.use_time_signatures:
                dic["Rest"] += ["TimeSig"]
                dic["TimeSig"] += ["Rest"]

        if self.config.use_programs:
            dic["Program"] = ["Pitch"]
            dic["Duration"] += ["Program"]
            dic["Duration"].remove("Pitch")
            dic["TimeShift"] += ["Program"]
            dic["TimeShift"].remove("Pitch")
            if self.config.use_chords:
                dic["Program"] += ["Chord"]
                dic["Chord"] += ["Program"]
                dic["Chord"].remove("Pitch")
            if self.config.use_tempos:
                dic["Tempo"] += ["Program"]
                dic["Tempo"].remove("Pitch")
            if self.config.use_rests:
                dic["Rest"] += ["Program"]
                dic["Rest"].remove("Pitch")

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Tempo":
            return 1
        elif x.type == "TimeSig":
            return 2
        elif x.type == "TimeShift" or x.type == "Rest":
            return 1000  # always last
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
