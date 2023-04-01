from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

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
)


class TSD(MIDITokenizer):
    r"""TSD, for Time Shift Duration, is similar to MIDI-Like :ref:`MIDI-Like`
    but uses explicit *Duration* tokens to represent note durations, which have
    showed `better results than with NoteOff tokens <https://arxiv.org/abs/2002.00212>`_.
    **Note:** as TSD uses *TimeShifts* events to move the time from note to
    note, it could be unsuited for tracks with long pauses. In such case, the
    maximum *TimeShift* value will be used.

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
        additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS,
        params: Union[str, Path] = None,
    ):
        additional_tokens["TimeSignature"] = False  # not compatible
        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            special_tokens,
            params=params,
        )

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self._current_midi_metadata["time_division"] / max(
            self.beat_res.values()
        )
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        min_rest = (
            self._current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.additional_tokens["Rest"]
            else 0
        )
        events = []

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Note On / Velocity / Duration
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
        # Adds tempo events if specified
        if self.additional_tokens["Tempo"]:
            for tempo_change in self._current_midi_metadata["tempo_changes"]:
                events.append(
                    Event(
                        type="Tempo",
                        value=tempo_change.tempo,
                        time=tempo_change.time,
                        desc=tempo_change.tempo,
                    )
                )

        # Sorts events
        events.sort(key=lambda x: x.time)

        # Time Shift
        previous_tick = 0
        previous_note_end = track.notes[0].start + 1
        for e, event in enumerate(events.copy()):
            # No time shift
            if event.time == previous_tick:
                pass

            # (Rest)
            elif (
                event.type in ["Pitch", "Tempo"]
                and self.additional_tokens["Rest"]
                and event.time - previous_note_end >= min_rest
            ):
                rest_beat, rest_pos = divmod(
                    event.time - previous_tick,
                    self._current_midi_metadata["time_division"],
                )
                rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                rest_pos = round(rest_pos / ticks_per_sample)
                rest_tick = previous_tick  # untouched tick value to the order is not messed after sorting

                if rest_beat > 0:
                    events.append(
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
                    events.append(
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
                    events.append(
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
                events.append(
                    Event(
                        type="TimeShift",
                        value=".".join(map(str, self.durations[index])),
                        time=previous_tick,
                        desc=f"{time_shift} ticks",
                    )
                )

            if event.type == "Pitch":
                previous_note_end = max(previous_note_end, event.desc)
            previous_tick = event.time

        # Adds chord events if specified
        if self.additional_tokens["Chord"] and not track.is_drum:
            events += detect_chords(
                track.notes,
                self._current_midi_metadata["time_division"],
                chord_maps=self.additional_tokens["chord_maps"],
                specify_root_note=self.additional_tokens["chord_tokens_with_root_note"],
                beat_res=self._first_beat_res,
                unknown_chords_nb_notes_range=self.additional_tokens["chord_unknown"],
            )

        events.sort(key=lambda x: (x.time, self._order(x)))

        return TokSequence(events=events)

    @_in_as_seq()
    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        ticks_per_sample = time_division // max(self.beat_res.values())
        tokens = tokens.tokens

        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(tokens):
            if tokens[ei].split("_")[0] == "Pitch":
                try:
                    if (
                        tokens[ei + 1].split("_")[0] == "Velocity"
                        and tokens[ei + 2].split("_")[0] == "Duration"
                    ):
                        pitch = int(tokens[ei].split("_")[1])
                        vel = int(tokens[ei + 1].split("_")[1])
                        duration = self._token_duration_to_ticks(
                            tokens[ei + 2].split("_")[1], time_division
                        )
                        instrument.notes.append(
                            Note(vel, pitch, current_tick, current_tick + duration)
                        )
                        ei += 1
                except IndexError:
                    pass
            elif tokens[ei].split("_")[0] == "TimeShift":
                current_tick += self._token_duration_to_ticks(
                    tokens[ei].split("_")[1], time_division
                )
            elif tokens[ei].split("_")[0] == "Rest":
                beat, pos = map(int, tokens[ei].split("_")[1].split("."))
                current_tick += beat * time_division + pos * ticks_per_sample
            elif tokens[ei].split("_")[0] == "Tempo":
                tempo = int(tokens[ei].split("_")[1])
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

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
        vocab += [f"Pitch_{i}" for i in self.pitch_range]

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
        if self.additional_tokens["Chord"]:
            vocab += self._create_chords_tokens()

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
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Pitch", "TimeShift"]
        dic["TimeShift"] = ["Pitch"]

        if self.additional_tokens["Chord"]:
            dic["Chord"] = ["Pitch"]
            dic["TimeShift"] += ["Chord"]

        if self.additional_tokens["Tempo"]:
            dic["TimeShift"] += ["Tempo"]
            dic["Tempo"] = ["Pitch", "TimeShift"]
            if self.additional_tokens["Chord"]:
                dic["Tempo"] += ["Chord"]

        if self.additional_tokens["Rest"]:
            dic["Rest"] = ["Rest", "Pitch", "TimeShift"]
            if self.additional_tokens["Chord"]:
                dic["Rest"] += ["Chord"]
            dic["Duration"] += ["Rest"]

        return dic

    def token_types_errors_training(
        self, x_tokens: List[int], y_tokens: List[int]
    ) -> Tuple[Union[float, Any]]:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a Pitch token should not be present if the same pitch is already being played

        :param x_tokens: input tokens
        :param y_tokens: produced token to check according to input
        :return: the error ratio (lower is better)
        """
        err_type = 0
        err_note = 0
        current_pitches = []

        for x_tok, y_tok in zip(x_tokens, y_tokens):
            x_type, x_value = self[x_tok].split("_")
            y_type, y_value = self[y_tok].split("_")
            if x_type == "PAD":
                break

            if x_type == "Pitch":
                current_pitches.append(int(x_value))
            elif x_type in ["TimeShift", "Rest"]:
                current_pitches = []  # moving in time, list reset

            # Good token type
            if y_type in self.tokens_types_graph[x_type]:
                if y_type == "Pitch" and int(y_value) in current_pitches:
                    err_note += 1  # pitch already being played
            # Bad token type
            else:
                err_type += 1

        return tuple(map(lambda err: err / len(x_tokens), (err_type, 0.0, err_note)))

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Program":
            return 0
        elif x.type == "Tempo":
            return 1
        elif x.type == "Chord":
            return 2
        elif x.type == "TimeShift" or x.type == "Rest":
            return 1000  # always last
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
