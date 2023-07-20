from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature

from ..classes import Event, TokSequence, TokenizerConfig
from ..constants import (
    MIDI_INSTRUMENTS,
    TEMPO,
    TIME_DIVISION,
    TIME_SIGNATURE,
)
from ..midi_tokenizer import MIDITokenizer, _in_as_seq, _out_as_complete_seq
from ..utils import detect_chords


class REMIPlus(MIDITokenizer):
    r"""REMI+ is extended REMI representation (Huang and Yang) for general
    multi-track, multi-signature symbolic music sequences, introduced in
    `FIGARO (Rütte et al.) <https://arxiv.org/abs/2201.10936>`, which
    represents notes as successions of *Program* (originally *Instrument* in the paper),
    *Pitch*, *Velocity* and *Duration* tokens, and time with *Bar* and *Position* tokens.
    A *Bar* token indicate that a new bar is beginning, and *Position* the current
    position within the current bar. The number of positions is determined by
    the ``beat_res`` argument, the maximum value will be used as resolution.

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
        super().__init__(tokenizer_config, True, params)

    def _tweak_config_before_creating_voc(self):
        self.config.use_programs = True
        # code handling rest decoding is writen, but not for detection (encoding)
        self.config.use_rests = False
        # self.unique_track = True

        # In case the tokenizer has been created without specifying any config or params file path
        if "max_bar_embedding" not in self.config.additional_params:
            # If used, this attribute might increase over tokenizations, if the tokenizer encounter longer MIDIs
            self.config.additional_params["max_bar_embedding"] = None

    def __notes_to_events(self, tracks: List[Instrument]) -> List[Event]:
        """Convert multi-track notes into one Token sequence.

        :param tracks: list of tracks (`miditoolkit.Instrument`) to convert.
        :return: sequences of Event.
        """
        # Flatten all notes
        notes_with_program = [
            (note, (track.program, track.is_drum))
            for track in tracks
            for note in track.notes
        ]
        notes_with_program.sort(key=lambda n: (n[0].start, n[0].pitch))

        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(self.config.beat_res.values())
        dur_bins = self._durations_ticks[self._current_midi_metadata["time_division"]]
        # Creates events
        events: List[Event] = []
        previous_tick = -1
        previous_note_end = (
            notes_with_program[0][0].start + 1
        )  # so that no rest is created before the first note
        # Bar
        if self.config.additional_params[
            "max_bar_embedding"
        ]:  # Check bar embedding limit, update if needed
            nb_bars = ceil(
                max(n[0].end for n in notes_with_program)
                / (self._current_midi_metadata["time_division"] * 4)
            )
            if self.config.additional_params["max_bar_embedding"] < nb_bars:
                for i in range(
                    self.config.additional_params["max_bar_embedding"], nb_bars
                ):
                    self.add_to_vocab(f"Bar_{i}")
                self.config.additional_params["max_bar_embedding"] = nb_bars
        current_bar = -1
        # Tempo
        current_tempo_idx = 0
        current_tempo = self._current_midi_metadata["tempo_changes"][
            current_tempo_idx
        ].tempo
        # TimeSignature
        current_time_sig_idx = 0
        current_time_sig_tick = 0
        current_time_sig_bar = 0
        time_sig_change = self._current_midi_metadata["time_sig_changes"][
            current_time_sig_idx
        ]
        current_time_sig = self._reduce_time_signature(
            time_sig_change.numerator, time_sig_change.denominator
        )
        ticks_per_bar = time_division * current_time_sig[0]
        # (Chord)
        if self.config.use_chords:  # "Chord" in additional tokens
            for track in tracks:  # find chords per track
                if track.is_drum:
                    continue
                chords = detect_chords(
                    track.notes,
                    self._current_midi_metadata["time_division"],
                    chord_maps=self.config.chord_maps,
                    specify_root_note=self.config.chord_tokens_with_root_note,
                    beat_res=self._first_beat_res,
                    unknown_chords_nb_notes_range=self.config.chord_unknown,
                )
                for chord in chords:
                    pos_index = int((chord.time % ticks_per_bar) / ticks_per_sample)
                    events.append(
                        Event("Position", pos_index, chord.time, "PositionChord")
                    )
                    events.append(
                        Event("Program", track.program, chord.time, "ProgramChord")
                    )
                    events.append(chord)
        events.sort(key=lambda x: x.time)

        for note, (program_num, is_drum) in notes_with_program:
            if note.start != previous_tick:
                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    events.append(
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
                current_bar += nb_new_bars

                # (TimeSignature)
                if self.config.use_time_signatures:
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
                if self.config.use_tempos:
                    is_tempo_changed = False
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
                                is_tempo_changed = True
                            else:  # <==> elif tempo_change.time > previous_tick:
                                break  # this tempo change is beyond the current time step, we break the loop
                    if is_tempo_changed or nb_new_bars > 0:  # after the new Bar token
                        # Position before the Tempo token
                        pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                        events.append(
                            Event(
                                type="Position",
                                value=pos_index,
                                time=note.start,
                                desc="PositionTempo",
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
        events = self.__notes_to_events([track])
        return TokSequence(events=events)  # type: ignore

    def tokens_to_track(
        self,
        tokens: TokSequence,
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
        events = self.__notes_to_events(midi.instruments)
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
        :param _: unused, to match parent method signature
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        tokens = cast(TokSequence, tokens)
        midi = MidiFile(ticks_per_beat=time_division)
        assert (
            time_division % max(self.config.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.config.beat_res.values())}"
        tokens = cast(List[str], tokens.tokens)  # for reducing type errors
        ticks_per_sample = time_division // max(self.config.beat_res.values())

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

        # TIME SIGNATURE
        if self.config.use_time_signatures:
            vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]

        # CHORD
        if self.config.use_chords:
            vocab += self._create_chords_tokens()

        # TEMPO
        if self.config.use_tempos:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        # PROGRAM
        vocab += [f"Program_{program}" for program in self.config.programs]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.

        :return: the token types transitions dictionary
        """
        dic: Dict[str, List[str]] = dict()

        dic["Bar"] = ["Position", "Bar"]
        dic["Position"] = ["Program"]
        dic["Program"] = ["Pitch", "Chord"]
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = ["Program", "Position", "Bar"]

        if self.config.use_time_signatures:
            dic["Bar"] = ["TimeSig", "Bar"]
            dic["TimeSig"] = ["Position"]

        if self.config.use_chords:
            dic["Chord"] = ["Position"]
            dic["Position"] += ["Chord"]

        if self.config.use_tempos:
            dic["Tempo"] = ["Position"]
            dic["Position"] += ["Tempo"]

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
        elif x.type == "Position" and x.desc == "PositionTempo":
            return 2
        elif x.type == "Tempo":
            return 3
        elif x.type == "Rest":
            return 7
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 8

    @_in_as_seq(complete=False, decode_bpe=False)
    def tokens_errors(
        self, tokens_to_check: Union[TokSequence, List[Union[int, List[int]]]]
    ) -> float:
        tokens_to_check = cast(TokSequence, tokens_to_check)
        nb_tok_predicted = len(tokens_to_check)  # used to norm the score
        if self.has_bpe:
            self.decode_bpe(tokens_to_check)
        self.complete_sequence(tokens_to_check)

        # Override from here
        tokens = cast(List[str], tokens_to_check.tokens)

        err_type = 0  # i.e. incompatible next type predicted
        err_time = 0  # i.e. goes back or stay in time (does not go forward)
        err_note = 0  # i.e. duplicated
        previous_type = tokens[0].split("_")[0]
        current_pos = -1
        current_program = 0
        current_pitches = {p: [] for p in self.config.programs}

        # Init first note and current pitches if needed
        if previous_type == "Pitch":
            pitch_val = int(tokens[0].split("_")[1])
            current_pitches[current_program].append(pitch_val)
        elif previous_type == "Position":
            current_pos = int(tokens[0].split("_")[1])

        for token in tokens[1:]:
            event_type, event_value = token.split("_")[0], token.split("_")[1]

            # Good token type
            if event_type in self.tokens_types_graph[previous_type]:
                if event_type == "Bar":  # reset
                    current_pos = -1
                    current_pitches = {p: [] for p in self.config.programs}
                elif previous_type == "Pitch":
                    pitch_val = int(event_value)
                    if pitch_val in current_pitches[current_program]:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches[current_program].append(pitch_val)
                elif event_type == "Position":
                    if int(event_value) < current_pos:
                        err_time += 1  # token position value <= to the current position
                    else:
                        current_pos = int(event_value)
                        current_pitches = {p: [] for p in self.config.programs}
                elif event_type == "Program":  # reset
                    current_program = int(event_value)
            # Bad token type
            else:
                err_type += 1
            previous_type = event_type

        return (err_type + err_time + err_note) / nb_tok_predicted
