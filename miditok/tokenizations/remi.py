from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature

from ..classes import Event, TokSequence, TokenizerConfig
from ..constants import (
    MIDI_INSTRUMENTS,
    TEMPO,
    TIME_DIVISION,
    TIME_SIGNATURE,
)
from ..midi_tokenizer import MIDITokenizer, _in_as_seq
from ..utils import detect_chords


class REMI(MIDITokenizer):
    r"""REMI, standing for Revamped MIDI and introduced with the
    `Pop Music Transformer (Huang and Yang) <https://dl.acm.org/doi/10.1145/3394171.3413671>`_,
    is a tokenization that represents notes as successions of *Pitch*, *Velocity* and *Duration*
    tokens, and time with *Bar* and *Position* tokens. A *Bar* token indicate that a new bar
    is beginning, and *Position* the current position within the current bar. The number of
    positions is determined by the ``beat_res`` argument, the maximum value will be used as
    resolution.
    With the `Program` and `TimeSignature` additional tokens enables, this class is equivalent to REMI+.
    REMI+ is an extended version of :ref:`REMI` (Huang and Yang) for general
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`, which handle multiple instruments by
    adding `Program` tokens before the `Pitch` ones.

    **NOTE:** in the original paper, the tempo information is represented as the succession
    of two token types: a *TempoClass* indicating if the tempo is fast or slow, and a
    *TempoValue* indicating its value. MidiTok only uses one *Tempo* token for its value
    (see :ref:`Additional tokens`).

    :param tokenizer_config: the tokenizer's configuration, as a :class:`miditok.classes.TokenizerConfig` object.
    :param max_bar_embedding: Maximum number of bars ("Bar_0", "Bar_1",...,"Bar_{num_bars-1}").
            If None passed, creates "Bar_None" token only in vocabulary for Bar token.
    :param params: path to a tokenizer config file. This will override other arguments and
            load the tokenizer based on the config file. This is particularly useful if the
            tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        max_bar_embedding: Optional[int] = None,
        params: Union[str, Path] = None,
    ):
        if (
            tokenizer_config is not None
            and "max_bar_embedding" not in tokenizer_config.additional_params
        ):
            # If used, this attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
            tokenizer_config.additional_params["max_bar_embedding"] = max_bar_embedding
        super().__init__(tokenizer_config, tokenizer_config.use_programs, params)

    def _tweak_config_before_creating_voc(self):
        # In case the tokenizer has been created without specifying any config or params file path
        if "max_bar_embedding" not in self.config.additional_params:
            # If used, this attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
            self.config.additional_params["max_bar_embedding"] = None

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
        current_bar = -1
        previous_tick = -1
        previous_note_end = 0
        current_time_sig = TIME_SIGNATURE
        ticks_per_bar = (
            self._current_midi_metadata["time_division"] * current_time_sig[0]
        )
        for e, event in enumerate(events):
            if event.type == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                ticks_per_bar = (
                    self._current_midi_metadata["time_division"] * current_time_sig[0]
                )
            if event.time != previous_tick:
                # (Rest)
                if (
                    event.type in ["Pitch", "Chord", "Tempo", "TimeSig"]
                    and self.config.use_rests
                    and event.time - previous_note_end >= min_rest
                ):
                    previous_tick = previous_note_end
                    rest_beat, rest_pos = divmod(
                        event.time - previous_tick,
                        self._current_midi_metadata["time_division"],
                    )
                    rest_beat = min(rest_beat, max([r[0] for r in self.rests]))
                    rest_pos = round(rest_pos / ticks_per_sample)

                    if rest_beat > 0:
                        all_events.append(
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
                        all_events.append(
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
                nb_new_bars = event.time // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    all_events.append(
                        Event(
                            type="Bar",
                            value=str(current_bar + i + 1)
                            if self.config.additional_params["max_bar_embedding"]
                            is not None
                            else "None",
                            time=(current_bar + i + 1) * ticks_per_bar,
                            desc=0,
                        )
                    )
                    if self.config.use_time_signatures:
                        all_events.append(
                            Event(
                                type="TimeSig",
                                value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                                time=(current_bar + i + 1) * ticks_per_bar,
                                desc=0,
                            )
                        )
                current_bar += nb_new_bars

                # Position
                pos_index = int((event.time % ticks_per_bar) / ticks_per_sample)
                all_events.append(
                    Event(
                        type="Position",
                        value=pos_index,
                        time=event.time,
                        desc=event.time,
                    )
                )

                previous_tick = event.time

            if event.type != "TimeSig":
                all_events.append(event)

            # Update max offset time of the notes encountered
            if event.type == "Pitch":
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type == "Tempo":
                previous_note_end = max(previous_note_end, event.time)

        # So sorting needed
        return all_events

    def tokens_to_track(
        self,
        tokens: TokSequence,
        time_division: Optional[int] = TIME_DIVISION,
        program: Optional[Tuple[int, bool]] = (0, False),
    ) -> Tuple[Instrument, List[TempoChange]]:
        pass

    def track_to_tokens(self, track: Instrument) -> TokSequence:
        pass

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
        if not self.one_token_stream:
            for i in range(len(midi.instruments)):
                all_events.append([])

        # Adds tempo events if specified
        if self.config.use_tempos:
            tempo_events = []
            for tempo_change in midi.tempo_changes:
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

        # Add time signature tokens if specified
        if self.config.use_time_signatures:
            time_sig_events = []
            for time_signature_change in midi.time_signature_changes:
                time_sig_events.append(
                    Event(
                        type="TimeSig",
                        value=f"{time_signature_change.numerator}/{time_signature_change.denominator}",
                        time=time_signature_change.time,
                    )
                )
            if self.one_token_stream:
                all_events += time_sig_events
            else:
                for i in range(len(all_events)):
                    all_events[i] += time_sig_events

        # Adds note tokens
        for ti, track in enumerate(midi.instruments):
            note_events = self.__notes_to_events(track)
            if self.one_token_stream:
                all_events += note_events
            else:
                all_events[ti] += note_events

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
        ticks_per_bar = time_division * TIME_SIGNATURE[0]  # init

        current_tick = 0
        current_bar = -1
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
                if token.split("_")[0] == "Bar":
                    current_bar += 1
                    current_tick = current_bar * ticks_per_bar
                elif token.split("_")[0] == "Rest":
                    beat, pos = map(int, seq[ti].split("_")[1].split("."))
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
                    if (
                        num != time_signature_changes[-1].numerator
                        and den != time_signature_changes[-1].denominator
                    ):
                        time_signature_changes.append(
                            TimeSignature(num, den, current_tick)
                        )
                        ticks_per_bar = (
                            self._current_midi_metadata["time_division"] * num
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

        # BAR
        if self.config.additional_params["max_bar_embedding"] is not None:
            vocab += [
                f"Bar_{i}"
                for i in range(self.config.additional_params["max_bar_embedding"])
            ]
        else:
            vocab += ["Bar_None"]

        # PITCH
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # VELOCITY
        vocab += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # POSITION
        nb_positions = max(self.config.beat_res.values()) * 4  # 4/4 time signature
        vocab += [f"Position_{i}" for i in range(nb_positions)]

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

        :return: the token types transitions dictionary
        """
        dic: Dict[str, List[str]] = dict()

        dic["Bar"] = ["Position", "Bar"]
        dic["Position"] = ["Pitch"]
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Pitch", "Position", "Bar"]

        if self.config.use_time_signatures:
            dic["Bar"] = ["TimeSig"]
            dic["TimeSig"] = ["Position", "Bar"]

        if self.config.use_chords:
            dic["Chord"] = ["Pitch"]
            dic["Position"] += ["Chord"]

        if self.config.use_tempos:
            dic["Tempo"] = ["Bar", "Position"]
            if self.config.use_programs:
                dic["Tempo"].append("Program")
            else:
                dic["Tempo"].append("Pitch")
            dic["Position"] += ["Tempo"]
            dic["Bar"] += ["Tempo"]
            if self.config.use_time_signatures:
                dic["TimeSig"].append("Tempo")

        if self.config.use_rests:
            dic["Rest"] = ["Rest", "Position", "Bar"]
            dic["Duration"] += ["Rest"]
            if self.config.use_time_signatures:
                dic["TimeSig"].append("Rest")
            if self.config.use_tempos:
                dic["Tempo"].append("Rest")

        if self.config.use_programs:
            dic["Program"] = ["Pitch", "Chord"]
            dic["Chord"] = ["Program"]
            dic["Position"].remove("Pitch")
            dic["Position"].append("Program")
            dic["Duration"].remove("Pitch")
            dic["Duration"].append("Program")

        return dic
