"""Structured MIDI encoding method as using in the Piano Inpainting Application
https://arxiv.org/abs/2107.05944

"""

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from .midi_tokenizer_base import MIDITokenizer
from .vocabulary import Vocabulary, Event
from .constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
)


class Structured(MIDITokenizer):
    r"""Structured MIDI encoding method as using in the Piano Inpainting Application
    https://arxiv.org/abs/2107.05944
    The token types follows the specific pattern:
    Pitch -> Velocity -> Duration -> Time Shift -> back to Pitch ...
    NOTE: this encoding uses only "Time Shifts" events to move in the time, and only
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
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo...), only programs
            are considered here.
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
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
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

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        events = []

        dur_bins = self.durations_ticks[self.current_midi_metadata["time_division"]]

        # First time shift if needed
        if track.notes[0].start != 0:
            if track.notes[0].start > max(dur_bins):
                time_shift = (
                    track.notes[0].start % self.current_midi_metadata["time_division"]
                )  # beat wise
            else:
                time_shift = track.notes[0].start
            index = np.argmin(np.abs(dur_bins - time_shift))
            events.append(
                Event(
                    type_="TimeShift",
                    value=".".join(map(str, self.durations[index])),
                    time=0,
                    desc=f"{time_shift} ticks",
                )
            )

        # Creates the Pitch, Velocity, Duration and Time Shift events
        for n, note in enumerate(track.notes[:-1]):
            # Pitch
            events.append(
                Event(type_="Pitch", value=note.pitch, time=note.start, desc=note.pitch)
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
            # Duration
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(
                Event(
                    type_="Duration",
                    value=".".join(map(str, self.durations[index])),
                    time=note.start,
                    desc=f"{duration} ticks",
                )
            )
            # TimeShift
            time_shift = track.notes[n + 1].start - note.start
            index = np.argmin(np.abs(dur_bins - time_shift))
            events.append(
                Event(
                    type_="TimeShift",
                    time=note.start,
                    desc=f"{time_shift} ticks",
                    value=".".join(map(str, self.durations[index]))
                    if time_shift != 0
                    else "0.0.1",
                )
            )
        # Adds the last note
        if track.notes[-1].pitch not in self.pitch_range:
            if len(events) > 0:
                del events[-1]
        else:
            events.append(
                Event(
                    type_="Pitch",
                    value=track.notes[-1].pitch,
                    time=track.notes[-1].start,
                    desc=track.notes[-1].pitch,
                )
            )
            events.append(
                Event(
                    type_="Velocity",
                    value=track.notes[-1].velocity,
                    time=track.notes[-1].start,
                    desc=f"{track.notes[-1].velocity}",
                )
            )
            duration = track.notes[-1].end - track.notes[-1].start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(
                Event(
                    type_="Duration",
                    value=".".join(map(str, self.durations[index])),
                    time=track.notes[-1].start,
                    desc=f"{duration} ticks",
                )
            )

        events.sort(key=lambda x: x.time)

        return self.events_to_tokens(events)

    def tokens_to_track(
        self,
        tokens: List[int],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and a "Dummy" tempo change
        """
        events = self.tokens_to_events(tokens)

        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        current_tick = 0
        count = 0

        while count < len(events):
            if events[count].type == "Pitch":
                if (
                    count + 2 < len(events)
                    and events[count + 1].type == "Velocity"
                    and events[count + 2].type == "Duration"
                ):
                    pitch = int(events[count].value)
                    vel = int(events[count + 1].value)
                    duration = self._token_duration_to_ticks(
                        events[count + 2].value, time_division
                    )
                    instrument.notes.append(
                        Note(vel, pitch, current_tick, current_tick + duration)
                    )
                    count += 3
                else:
                    count += 1
            elif events[count].type == "TimeShift":
                beat, pos, res = map(int, events[count].value.split("."))
                current_tick += (beat * res + pos) * time_division // res  # time shift
                count += 1
            else:
                count += 1

        return instrument, [TempoChange(TEMPO, 0)]

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: DEPRECIATED, will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        if sos_eos_tokens is not None:
            print(
                "\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, "
                "_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m"
            )
        vocab = Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)

        # PITCH
        vocab.add_event(f"Pitch_{i}" for i in self.pitch_range)

        # VELOCITY
        vocab.add_event(f"Velocity_{i}" for i in self.velocities)

        # DURATION
        vocab.add_event(
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        )

        # TIME SHIFT (same as durations)
        vocab.add_event("TimeShift_0.0.1")  # for a time shift of 0
        vocab.add_event(
            f'TimeShift_{".".join(map(str, duration))}' for duration in self.durations
        )

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
        dic = {
            "Pitch": ["Velocity"],
            "Velocity": ["Duration"],
            "Duration": ["TimeShift"],
            "TimeShift": ["Pitch"],
        }
        self._add_special_tokens_to_types_graph(dic)
        return dic

    def token_types_errors(
        self, tokens: List[int], consider_pad: bool = False
    ) -> float:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch values are also analyzed:
            - a pitch token should not be present if the same pitch is already played at the time

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        nb_tok_predicted = len(tokens)  # used to norm the score
        tokens = self.decompose_bpe(tokens) if self.has_bpe else tokens

        # Override from here

        err = 0
        previous_type = self.vocab.token_type(tokens[0])
        current_pitches = []

        def check(tok: int):
            nonlocal err, previous_type, current_pitches
            token_type, token_value = self.vocab.token_to_event[tok].split("_")

            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == "Pitch":
                    if int(token_value) in current_pitches:
                        err += 1  # pitch already played at current position
                    else:
                        current_pitches.append(int(token_value))
                elif token_type == "TimeShift":
                    if self._token_duration_to_ticks(token_value, 48) > 0:
                        current_pitches = []  # moving in time, list reset
            # Bad token type
            else:
                err += 1
            previous_type = token_type

        if consider_pad:
            for token in tokens[1:]:
                check(token)
        else:
            for token in tokens[1:]:
                if previous_type == "PAD":
                    break
                check(token)
        return err / nb_tok_predicted
