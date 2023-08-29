from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
from miditoolkit import (
    MidiFile,
    Instrument,
    Note,
    TempoChange,
    TimeSignature,
    Pedal,
    PitchBend,
)

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
        # Add time events
        all_events = []
        previous_tick = 0
        previous_note_end = 0
        for e, event in enumerate(events):
            # No time shift
            if event.time != previous_tick:
                # (Rest)
                if (
                    self.config.use_rests
                    and event.time - previous_note_end >= self._min_rest
                ):
                    previous_tick = previous_note_end
                    rest_values = self._ticks_to_duration_tokens(
                        event.time - previous_tick, rest=True
                    )
                    for dur_value, dur_ticks in zip(*rest_values):
                        all_events.append(
                            Event(
                                type="Rest",
                                value=".".join(map(str, dur_value)),
                                time=previous_tick,
                                desc=f"{event.time - previous_tick} ticks",
                            )
                        )
                        previous_tick += dur_ticks

                # Time shift
                # no else here as previous might have changed with rests
                if event.time != previous_tick:
                    time_shift = event.time - previous_tick
                    for dur_value, dur_ticks in zip(
                        *self._ticks_to_duration_tokens(time_shift)
                    ):
                        all_events.append(
                            Event(
                                type="TimeShift",
                                value=".".join(map(str, dur_value)),
                                time=previous_tick,
                                desc=f"{time_shift} ticks",
                            )
                        )
                        previous_tick += dur_ticks
                previous_tick = event.time

            all_events.append(event)

            # Update max offset time of the notes encountered
            if event.type == "Pitch":
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type in [
                "Program",
                "Tempo",
                "Pedal",
                "PedalOff",
                "PitchBend",
                "Chord",
            ]:
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
        active_pedals = {}

        def check_inst(prog: int):
            if prog not in instruments.keys():
                instruments[prog] = Instrument(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

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
                tok_type, tok_val = token.split("_")
                if tok_type in ["TimeShift", "Rest"]:
                    if tok_type == "Rest":
                        current_tick = max(previous_note_end, current_tick)
                    current_tick += self._token_duration_to_ticks(
                        tok_val, time_division
                    )
                elif tok_type == "Pitch":
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
                            check_inst(current_program)
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
                elif tok_type == "Program":
                    current_program = int(tok_val)
                elif tok_type == "Tempo":
                    # If your encoding include tempo tokens, each Position token should be followed by
                    # a tempo token, but if it is not the case this method will skip this step
                    tempo = float(tok_val)
                    if si == 0 and current_tick != tempo_changes[-1].time:
                        tempo_changes.append(TempoChange(tempo, current_tick))
                    previous_note_end = max(previous_note_end, current_tick)
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    current_time_signature = time_signature_changes[-1]
                    if (
                        si == 0
                        and num != current_time_signature.numerator
                        and den != current_time_signature.denominator
                    ):
                        time_signature_changes.append(
                            TimeSignature(num, den, current_tick)
                        )
                elif tok_type == "Pedal":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if self.config.sustain_pedal_duration and ti + 1 < len(seq):
                        if seq[ti + 1].split("_")[0] == "Duration":
                            duration = self._token_duration_to_ticks(
                                seq[ti + 1].split("_")[1], time_division
                            )
                            # Add instrument if it doesn't exist, can happen for the first tokens
                            check_inst(pedal_prog)
                            instruments[pedal_prog].pedals.append(
                                Pedal(current_tick, current_tick + duration)
                            )
                    else:
                        if pedal_prog not in active_pedals:
                            active_pedals[pedal_prog] = current_tick
                elif tok_type == "PedalOff":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if pedal_prog in active_pedals:
                        check_inst(pedal_prog)
                        instruments[pedal_prog].pedals.append(
                            Pedal(active_pedals[pedal_prog], current_tick)
                        )
                        del active_pedals[pedal_prog]
                elif tok_type == "PitchBend":
                    if current_program not in instruments.keys():
                        check_inst(current_program)
                    instruments[current_program].pitch_bends.append(
                        PitchBend(int(tok_val), current_tick)
                    )
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        if len(time_signature_changes) > 1:
            del time_signature_changes[0]  # delete mocked time signature change
        time_signature_changes[0].time = 0

        # create MidiFile
        midi.instruments = list(instruments.values())
        for instrument in midi.instruments:
            if instrument.program == -1:
                instrument.program = 0
                instrument.is_drum = True
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

        if self.config.use_programs:
            first_note_token_type = "Program"
            dic["Program"] = ["Pitch"]
        else:
            first_note_token_type = "Pitch"
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = [first_note_token_type, "TimeShift"]
        dic["TimeShift"] = [first_note_token_type]

        if self.config.use_chords:
            dic["Chord"] = [first_note_token_type]
            dic["TimeShift"] += ["Chord"]
            if self.config.use_programs:
                dic["Program"].append("Chord")

        if self.config.use_tempos:
            dic["TimeShift"] += ["Tempo"]
            dic["Tempo"] = [first_note_token_type, "TimeShift"]
            if self.config.use_chords:
                dic["Tempo"] += ["Chord"]
            if self.config.use_rests:
                dic["Tempo"].append("Rest")  # only for first token

        if self.config.use_time_signatures:
            dic["TimeShift"] += ["TimeSig"]
            dic["TimeSig"] = [first_note_token_type, "TimeShift"]
            if self.config.use_chords:
                dic["TimeSig"] += ["Chord"]
            if self.config.use_rests:
                dic["TimeSig"].append("Rest")  # only for first token
            if self.config.use_tempos:
                dic["Tempo"].append("TimeSig")

        if self.config.use_sustain_pedals:
            dic["TimeShift"].append("Pedal")
            if self.config.sustain_pedal_duration:
                dic["Pedal"] = ["Duration"]
                dic["Duration"] = [first_note_token_type, "TimeShift"]
            else:
                dic["PedalOff"] = [
                    "Pedal",
                    "PedalOff",
                    first_note_token_type,
                    "TimeShift",
                ]
                dic["Pedal"] = ["Pedal", first_note_token_type, "TimeShift"]
                dic["TimeShift"].append("PedalOff")
            if self.config.use_chords:
                dic["Pedal"].append("Chord")
                if not self.config.sustain_pedal_duration:
                    dic["PedalOff"].append("Chord")
                    dic["Chord"].append("PedalOff")
            if self.config.use_rests:
                dic["Pedal"].append("Rest")
                if not self.config.sustain_pedal_duration:
                    dic["PedalOff"].append("Rest")
            if self.config.use_tempos:
                dic["Tempo"].append("Pedal")
                if not self.config.sustain_pedal_duration:
                    dic["Tempo"].append("PedalOff")
            if self.config.use_time_signatures:
                dic["TimeSig"].append("Pedal")
                if not self.config.sustain_pedal_duration:
                    dic["TimeSig"].append("PedalOff")

        if self.config.use_pitch_bends:
            # As a Program token will precede PitchBend otherwise
            # Else no need to add Program as its already in
            dic["PitchBend"] = [first_note_token_type, "TimeShift"]
            if not self.config.use_programs:
                dic["TimeShift"].append("PitchBend")
                if self.config.use_tempos:
                    dic["Tempo"].append("PitchBend")
                if self.config.use_time_signatures:
                    dic["TimeSig"].append("PitchBend")
                if self.config.use_sustain_pedals:
                    dic["Pedal"].append("PitchBend")
                    if self.config.sustain_pedal_duration:
                        dic["Duration"].append("PitchBend")
                    else:
                        dic["PedalOff"].append("PitchBend")
            else:
                dic["Program"].append("PitchBend")
            if self.config.use_chords:
                dic["PitchBend"].append("Chord")
            if self.config.use_rests:
                dic["PitchBend"].append("Rest")

        if self.config.use_rests:
            dic["Rest"] = ["Rest", first_note_token_type, "TimeShift"]
            dic["Duration"].append("Rest")
            if self.config.use_chords:
                dic["Rest"] += ["Chord"]
            if self.config.use_tempos:
                dic["Rest"].append("Tempo")
            if self.config.use_time_signatures:
                dic["Rest"].append("TimeSig")
            if self.config.use_sustain_pedals:
                dic["Rest"].append("Pedal")
                if self.config.sustain_pedal_duration:
                    dic["Duration"].append("Rest")
                else:
                    dic["Rest"].append("PedalOff")
                    dic["PedalOff"].append("Rest")
            if self.config.use_pitch_bends:
                dic["Rest"].append("PitchBend")

        return dic
