from math import ceil
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

    **Note:** in the original paper, the tempo information is represented as the succession
    of two token types: a *TempoClass* indicating if the tempo is fast or slow, and a
    *TempoValue* indicating its value. MidiTok only uses one *Tempo* token for its value
    (see :ref:`Additional tokens`).
    **Note:** When decoding multiple token sequences (of multiple tracks), i.e. when `config.use_programs` is False,
    only the tempos and time signatures of the first sequence will be decoded for the whole MIDI.

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
        one_stream = (
            tokenizer_config.use_programs if tokenizer_config is not None else False
        )
        super().__init__(tokenizer_config, one_stream, params)

    def _tweak_config_before_creating_voc(self):
        # In case the tokenizer has been created without specifying any config or params file path
        if "max_bar_embedding" not in self.config.additional_params:
            # If used, this attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
            self.config.additional_params["max_bar_embedding"] = None

    def _add_time_events(self, events: List[Event]) -> List[Event]:
        r"""
        Takes a sequence of note events (containing optionally Chord, Tempo and TimeSignature tokens),
        and insert (not inplace) time tokens (TimeShift, Rest) to complete the sequence.

        :param events: note events to complete.
        :return: the same events, with time events inserted.
        """
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(self.config.beat_res.values())
        min_rest = (
            time_division * self.rests[0][0] + ticks_per_sample * self.rests[0][1]
            if self.config.use_rests
            else 0
        )

        # Add time events
        all_events = []
        current_bar = -1
        previous_tick = -1
        previous_note_end = 0
        current_time_sig = TIME_SIGNATURE
        ticks_per_bar = self._compute_ticks_per_bar(
            TimeSignature(*current_time_sig, 0), time_division
        )
        for e, event in enumerate(events):
            if event.type == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                ticks_per_bar = self._compute_ticks_per_bar(
                    TimeSignature(*current_time_sig, event.time), time_division
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
                        time_division,
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
                        previous_tick += rest_beat * time_division

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
        ticks_per_bar = self._compute_ticks_per_bar(
            time_signature_changes[0], time_division
        )  # init

        current_tick = 0
        current_bar = -1
        current_program = 0
        previous_note_end = 0
        for si, seq in enumerate(tokens):
            # Set track / sequence program if needed
            if not self.one_token_stream:
                current_tick = 0
                current_bar = -1
                ticks_per_bar = self._compute_ticks_per_bar(time_signature_changes[0], time_division)
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
                    tempo = float(token.split("_")[1])
                    if si == 0 and current_tick != tempo_changes[-1].time:
                        tempo_changes.append(TempoChange(tempo, current_tick))
                    previous_note_end = max(previous_note_end, current_tick)
                elif token.split("_")[0] == "TimeSig":
                    num, den = self._parse_token_time_signature(token.split("_")[1])
                    if (
                        num != time_signature_changes[-1].numerator
                        and den != time_signature_changes[-1].denominator
                    ):
                        time_sig = TimeSignature(num, den, current_tick)
                        if si == 0:
                            time_signature_changes.append(time_sig)
                        ticks_per_bar = self._compute_ticks_per_bar(
                            time_sig, time_division
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
        max_nb_beats = max(
            map(lambda ts: ceil(4 * ts[0] / ts[1]), self.time_signatures)
        )
        nb_positions = max(self.config.beat_res.values()) * max_nb_beats
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
