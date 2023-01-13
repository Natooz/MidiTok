"""MIDI-like encoding method similar to ???
Music Transformer:

"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer
from .vocabulary import Vocabulary, Event
from .utils import detect_chords
from .constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
    CHORD_MAPS,
)


class MIDILike(MIDITokenizer):
    r"""MIDI-Like encoding, used with Music Transformer or MT3
    https://arxiv.org/abs/1808.03715
    This strategy simply convert MIDI messages into distinct tokens.
    The token types are then NoteOn, Velocity, NoteOff and TimeShift
    (+ the additional token types of MidiTok if desired).
    NOTE: as this encoding uses only "Time Shifts" events to move in the time, and only
    from one note to another. Hence it is suitable to encode continuous sequences of
    notes without long periods of silence. If your dataset contains music with long
    pauses, you might handle them with an appropriate "time shift" dictionary
    (which values are made from the beat_res dict) or with a different encoding.

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo...)
    :param pad: will include a PAD token, used when training a model with batch of sequences of
            unequal lengths, and usually at index 0 of the vocabulary. (default: True)
    :param sos_eos: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary.
            (default: False)
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """

    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        additional_tokens["TimeSignature"] = False  # not compatible
        super().__init__(
            pitch_range,
            beat_res,
            nb_velocities,
            additional_tokens,
            pad,
            sos_eos,
            mask,
            params=params,
        )

    def track_to_tokens(self, track: Instrument) -> List[int]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self.current_midi_metadata["time_division"] / max(
            self.beat_res.values()
        )
        dur_bins = self.durations_ticks[self.current_midi_metadata["time_division"]]
        min_rest = (
            self.current_midi_metadata["time_division"] * self.rests[0][0]
            + ticks_per_sample * self.rests[0][1]
            if self.additional_tokens["Rest"]
            else 0
        )
        events = []

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Note On
            events.append(
                Event(type_="NoteOn", value=note.pitch, time=note.start, desc=note.end)
            )
            # Velocity
            events.append(
                Event(
                    type_="Velocity",
                    value=note.velocity,
                    time=note.start,
                    desc=f"{note.velocity}",
                )
            )
            # Note Off
            events.append(
                Event(type_="NoteOff", value=note.pitch, time=note.end, desc=note.end)
            )
        # Adds tempo events if specified
        if self.additional_tokens["Tempo"]:
            for tempo_change in self.current_midi_metadata["tempo_changes"]:
                events.append(
                    Event(
                        type_="Tempo",
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
                self.additional_tokens["Rest"]
                and event.type in ["NoteOn", "Tempo"]
                and event.time - previous_note_end >= min_rest
            ):
                rest_beat, rest_pos = divmod(
                    event.time - previous_tick,
                    self.current_midi_metadata["time_division"],
                )
                rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                rest_pos = round(rest_pos / ticks_per_sample)
                rest_tick = previous_tick  # untouched tick value to the order is not messed after sorting

                if rest_beat > 0:
                    events.append(
                        Event(
                            type_="Rest",
                            value=f"{rest_beat}.0",
                            time=rest_tick,
                            desc=f"{rest_beat}.0",
                        )
                    )
                    previous_tick += (
                        rest_beat * self.current_midi_metadata["time_division"]
                    )

                while rest_pos >= self.rests[0][1]:
                    rest_pos_temp = min(
                        [r[1] for r in self.rests], key=lambda x: abs(x - rest_pos)
                    )
                    events.append(
                        Event(
                            type_="Rest",
                            value=f"0.{rest_pos_temp}",
                            time=rest_tick,
                            desc=f"0.{rest_pos_temp}",
                        )
                    )
                    previous_tick += round(rest_pos_temp * ticks_per_sample)
                    rest_pos -= rest_pos_temp

                # Adds a time shift if needed
                if rest_pos > 0:
                    time_shift = round(rest_pos * ticks_per_sample)
                    index = np.argmin(np.abs(dur_bins - time_shift))
                    events.append(
                        Event(
                            type_="TimeShift",
                            value=".".join(map(str, self.durations[index])),
                            time=previous_tick,
                            desc=f"{time_shift} ticks",
                        )
                    )

            # TimeShift
            else:
                time_shift = event.time - previous_tick
                index = np.argmin(np.abs(dur_bins - time_shift))
                events.append(
                    Event(
                        type_="TimeShift",
                        value=".".join(map(str, self.durations[index])),
                        time=previous_tick,
                        desc=f"{time_shift} ticks",
                    )
                )

            if event.type == "NoteOn":
                previous_note_end = max(previous_note_end, event.desc)
            previous_tick = event.time

        # Adds chord events if specified
        if self.additional_tokens["Chord"] and not track.is_drum:
            events += detect_chords(
                track.notes,
                self.current_midi_metadata["time_division"],
                self._first_beat_res,
            )

        events.sort(key=lambda x: (x.time, self._order(x)))

        return self.events_to_tokens(events)

    def tokens_to_track(
        self,
        tokens: List[int],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
        default_duration: int = None,
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :param default_duration: default duration (in ticks) in case a Note On event occurs without its associated
                                note off event. Leave None to discard Note On with no Note Off event.
        :return: the miditoolkit instrument object and tempo changes
        """
        ticks_per_sample = time_division // max(self.beat_res.values())
        events = self.tokens_to_events(tokens)

        max_duration = self.durations[-1][0] * time_division + self.durations[-1][1] * (
            time_division // self.durations[-1][2]
        )
        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [
            TempoChange(TEMPO, -1)
        ]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(events):
            if events[ei].type == "NoteOn":
                try:
                    if events[ei + 1].type == "Velocity":
                        pitch = int(events[ei].value)
                        vel = int(events[ei + 1].value)

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        duration = 0
                        for i in range(ei + 1, len(events)):
                            if (
                                events[i].type == "NoteOff"
                                and int(events[i].value) == pitch
                            ):
                                duration = offset_tick
                                break
                            elif events[i].type == "TimeShift":
                                offset_tick += self._token_duration_to_ticks(
                                    events[i].value, time_division
                                )
                            elif events[ei].type == "Rest":
                                beat, pos = map(int, events[ei].value.split("."))
                                offset_tick += (
                                    beat * time_division + pos * ticks_per_sample
                                )
                            if (
                                offset_tick > max_duration
                            ):  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration != 0:
                            instrument.notes.append(
                                Note(vel, pitch, current_tick, current_tick + duration)
                            )
                        ei += 1
                except IndexError:
                    pass
            elif events[ei].type == "TimeShift":
                current_tick += self._token_duration_to_ticks(
                    events[ei].value, time_division
                )
            elif events[ei].type == "Rest":
                beat, pos = map(int, events[ei].value.split("."))
                current_tick += beat * time_division + pos * ticks_per_sample
            elif events[ei].type == "Tempo":
                tempo = int(events[ei].value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is used as a padding index for training.

        :param sos_eos_tokens: DEPRECIATED, will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        if sos_eos_tokens is not None:
            print(
                "\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, "
                "_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m"
            )
        vocab = Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)

        # NOTE ON
        vocab.add_event(f"NoteOn_{i}" for i in self.pitch_range)

        # NOTE OFF
        vocab.add_event(f"NoteOff_{i}" for i in self.pitch_range)

        # VELOCITY
        vocab.add_event(f"Velocity_{i}" for i in self.velocities)

        # TIME SHIFTS
        vocab.add_event(
            f'TimeShift_{".".join(map(str, duration))}' for duration in self.durations
        )

        # CHORD
        if self.additional_tokens["Chord"]:
            vocab.add_event(
                f"Chord_{i}" for i in range(3, 6)
            )  # non recognized chords (between 3 and 5 notes only)
            vocab.add_event(f"Chord_{chord_quality}" for chord_quality in CHORD_MAPS)

        # REST
        if self.additional_tokens["Rest"]:
            vocab.add_event(f'Rest_{".".join(map(str, rest))}' for rest in self.rests)

        # TEMPO
        if self.additional_tokens["Tempo"]:
            vocab.add_event(f"Tempo_{i}" for i in self.tempos)

        # PROGRAM
        if self.additional_tokens["Program"]:
            vocab.add_event(f"Program_{program}" for program in range(-1, 128))

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["NoteOn"] = ["Velocity"]
        dic["Velocity"] = ["NoteOn", "TimeShift"]
        dic["TimeShift"] = ["NoteOff", "NoteOn"]
        dic["NoteOff"] = ["NoteOff", "NoteOn", "TimeShift"]

        if self.additional_tokens["Chord"]:
            dic["Chord"] = ["NoteOn"]
            dic["TimeShift"] += ["Chord"]
            dic["NoteOff"] += ["Chord"]

        if self.additional_tokens["Tempo"]:
            dic["TimeShift"] += ["Tempo"]
            dic["Tempo"] = ["NoteOn", "TimeShift"]
            if self.additional_tokens["Chord"]:
                dic["Tempo"] += ["Chord"]

        if self.additional_tokens["Rest"]:
            dic["Rest"] = ["Rest", "NoteOn", "TimeShift"]
            if self.additional_tokens["Chord"]:
                dic["Rest"] += ["Chord"]
            dic["NoteOff"] += ["Rest"]

        self._add_special_tokens_to_types_graph(dic)
        return dic

    def token_types_errors(
        self, tokens: List[int], consider_pad: bool = False
    ) -> float:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a NoteOn token should not be present if the same pitch is already being played
            - a NoteOff token should not be present the note is not being played

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        nb_tok_predicted = len(tokens)  # used to norm the score
        tokens = self.decompose_bpe(tokens) if self.has_bpe else tokens

        # Override from here

        err = 0
        current_pitches = []
        max_duration = self.durations[-1][0] * max(self.beat_res.values())
        max_duration += self.durations[-1][1] * (
            max(self.beat_res.values()) // self.durations[-1][2]
        )

        events = self.tokens_to_events(tokens)

        for i in range(1, len(events)):
            # Good token type
            if events[i].type in self.tokens_types_graph[events[i - 1].type]:
                if events[i].type == "NoteOn":
                    if int(events[i].value) in current_pitches:
                        err += 1  # pitch already being played
                        continue

                    current_pitches.append(int(events[i].value))
                    # look for an associated note off event to get duration
                    offset_sample = 0
                    for j in range(i + 1, len(events)):
                        if events[j].type == "NoteOff" and int(events[j].value) == int(
                            events[i].value
                        ):
                            break  # all good
                        elif events[j].type == "TimeShift":
                            offset_sample += self._token_duration_to_ticks(
                                events[j].value, max(self.beat_res.values())
                            )

                        if (
                            offset_sample > max_duration
                        ):  # will not look for Note Off beyond
                            err += 1
                            break
                elif events[i].type == "NoteOff":
                    if int(events[i].value) not in current_pitches:
                        err += 1  # this pitch wasn't being played
                    else:
                        current_pitches.remove(int(events[i].value))
                elif not consider_pad and events[i].type == "PAD":
                    break

            # Bad token type
            else:
                err += 1

        return err / nb_tok_predicted

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Program":
            return 0
        elif x.type == "NoteOff":
            return 1
        elif x.type == "Tempo":
            return 2
        elif x.type == "Chord":
            return 3
        elif x.type == "TimeShift" or x.type == "Rest":
            return 1000  # always last
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
