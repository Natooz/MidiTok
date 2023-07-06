from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from miditoolkit import Instrument, Note, TempoChange

from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..classes import TokSequence, Event
from ..constants import (
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
)


class Structured(MIDITokenizer):
    r"""Introduced with the `Piano Inpainting Application <https://arxiv.org/abs/2002.00212>`_,
    it is similar to :ref:`TSD` but is based on a consistent token type successions.
    Token types always follow the same pattern: *Pitch* -> *Velocity* -> *Duration* -> *TimeShift*.
    The latter is set to 0 for simultaneous notes.
    To keep this property, no additional token can be inserted in MidiTok's implementation,
    except *Program* that can be added to the vocabulary and are up to you to use.
    **Note:** as Structured uses *TimeShifts* events to move the time from note to
    note, it could be unsuited for tracks with long pauses. In such case, the
    maximum *TimeShift* value will be used.
    """

    def _tweak_config_before_creating_voc(self):
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_tempos = False
        self.config.use_time_signatures = False

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens (:class:`miditok.TokSequence`).

        :param track: MIDI track to convert
        :return: :class:`miditok.TokSequence` of corresponding tokens.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        events = []

        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]

        # First time shift if needed
        if track.notes[0].start != 0:
            if track.notes[0].start > max(dur_bins):
                time_shift = (
                    track.notes[0].start % self._current_midi_metadata["time_division"]
                )  # beat wise
            else:
                time_shift = track.notes[0].start
            index = np.argmin(np.abs(dur_bins - time_shift))
            events.append(
                Event(
                    type="TimeShift",
                    value=".".join(map(str, self.durations[index])),
                    time=0,
                    desc=f"{time_shift} ticks",
                )
            )

        # Creates the Pitch, Velocity, Duration and Time Shift events
        for n, note in enumerate(track.notes[:-1]):
            # Pitch
            events.append(
                Event(type="Pitch", value=note.pitch, time=note.start, desc=note.pitch)
            )
            # Velocity
            events.append(
                Event(
                    type="Velocity",
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
                    type="Duration",
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
                    type="TimeShift",
                    time=note.start,
                    desc=f"{time_shift} ticks",
                    value=".".join(map(str, self.durations[index]))
                    if time_shift != 0
                    else "0.0.1",
                )
            )
        # Adds the last note
        if track.notes[-1].pitch not in range(*self.config.pitch_range):
            if len(events) > 0:
                del events[-1]
        else:
            events.append(
                Event(
                    type="Pitch",
                    value=track.notes[-1].pitch,
                    time=track.notes[-1].start,
                    desc=track.notes[-1].pitch,
                )
            )
            events.append(
                Event(
                    type="Velocity",
                    value=track.notes[-1].velocity,
                    time=track.notes[-1].start,
                    desc=f"{track.notes[-1].velocity}",
                )
            )
            duration = track.notes[-1].end - track.notes[-1].start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(
                Event(
                    type="Duration",
                    value=".".join(map(str, self.durations[index])),
                    time=track.notes[-1].start,
                    desc=f"{duration} ticks",
                )
            )

        events.sort(key=lambda x: x.time)

        return TokSequence(events=events)

    @_in_as_seq()
    def tokens_to_track(
        self,
        tokens: Union[TokSequence, List, np.ndarray, Any],
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object.

        :param tokens: sequence of tokens to convert. Can be either a Tensor (PyTorch and Tensorflow are supported),
                a numpy array, a Python list or a TokSequence.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and a "Dummy" tempo change
        """
        tokens = tokens.tokens

        name = "Drums" if program[1] else MIDI_INSTRUMENTS[program[0]]["name"]
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        current_tick = 0
        count = 0

        while count < len(tokens):
            if tokens[count].split("_")[0] == "Pitch":
                if (
                    count + 2 < len(tokens)
                    and tokens[count + 1].split("_")[0] == "Velocity"
                    and tokens[count + 2].split("_")[0] == "Duration"
                ):
                    pitch = int(tokens[count].split("_")[1])
                    vel = int(tokens[count + 1].split("_")[1])
                    duration = self._token_duration_to_ticks(
                        tokens[count + 2].split("_")[1], time_division
                    )
                    instrument.notes.append(
                        Note(vel, pitch, current_tick, current_tick + duration)
                    )
                    count += 3
                else:
                    count += 1
            elif tokens[count].split("_")[0] == "TimeShift":
                beat, pos, res = map(int, tokens[count].split("_")[1].split("."))
                current_tick += (beat * res + pos) * time_division // res  # time shift
                count += 1
            else:
                count += 1

        return instrument, [TempoChange(TEMPO, 0)]

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
        if sos_eos_tokens is not None:
            print(
                "\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, "
                "_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m"
            )
        vocab = []

        # PITCH
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # VELOCITY
        vocab += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # TIME SHIFT (same as durations)
        vocab.append("TimeShift_0.0.1")  # for a time shift of 0
        vocab += [
            f'TimeShift_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # PROGRAM
        if self.config.use_programs:
            vocab += [f"Program_{program}" for program in self.config.programs]

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
        return dic
