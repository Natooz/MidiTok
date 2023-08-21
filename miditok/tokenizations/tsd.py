from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from ..midi_tokenizer import MIDITokenizer, _in_as_seq
from ..classes import TokSequence, Event
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
    If you specify `use_programs` as `True` in the config file, the tokenizer will add `Program` tokens before
    each `Pitch` tokens to specify its instrument, and will treat all tracks as a single stream of tokens.

    **Note:** as `TSD` uses *TimeShifts* events to move the time from note to
    note, it can be unsuited for tracks with pauses longer than the maximum `TimeShift` value. In such cases, the
    maximum *TimeShift* value will be used.
    **Note:** When decoding multiple token sequences (of multiple tracks), i.e. when `config.use_programs` is False,
    only the tempos and time signatures of the first sequence will be decoded for the whole MIDI.
    """

    def _tweak_config_before_creating_voc(self):
        if self.config.use_programs:
            self.one_token_stream = True

    def _add_time_events(self, events: List[Event]) -> List[Event]:
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
        all_events = []
        previous_tick = 0
        previous_note_end = events[0].time + 1
        for e, event in enumerate(events):
            # No time shift
            if event.time != previous_tick:
                # (Rest)
                if (
                    event.type in ["Pitch", "Chord", "Tempo", "TimeSig"]
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

            all_events.append(event)

            # Update max offset time of the notes encountered
            if event.type == "Pitch":
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type == "Tempo":
                previous_note_end = max(previous_note_end, event.time)

        return all_events

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
                    tempo = float(token.split("_")[1])
                    if si == 0 and current_tick != tempo_changes[-1].time:
                        tempo_changes.append(TempoChange(tempo, current_tick))
                    previous_note_end = max(previous_note_end, current_tick)
                elif token.split("_")[0] == "TimeSig":
                    num, den = self._parse_token_time_signature(token.split("_")[1])
                    current_time_signature = time_signature_changes[-1]
                    if (
                        si == 0
                        and num != current_time_signature.numerator
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

    def _create_base_vocabulary(self) -> List[str]:
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

        # Add additional tokens
        self._add_additional_tokens_to_vocab_list(vocab)

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

        if self.config.use_time_signatures:
            dic["TimeShift"] += ["TimeSig"]
            dic["TimeSig"] = ["Pitch", "TimeShift"]
            if self.config.use_chords:
                dic["TimeSig"] += ["Chord"]
            if self.config.use_tempos:
                dic["Tempo"].append("TimeSig")

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
            if self.config.use_tempos:
                dic["TimeSig"] += ["Program"]
                dic["TimeSig"].remove("Pitch")
            if self.config.use_rests:
                dic["Rest"] += ["Program"]
                dic["Rest"].remove("Pitch")

        return dic
