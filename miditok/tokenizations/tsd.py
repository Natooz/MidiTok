from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from ..classes import Event, TokSequence
from ..constants import (
    MIDI_INSTRUMENTS,
    TIME_DIVISION,
)
from ..midi_tokenizer import MIDITokenizer


class TSD(MIDITokenizer):
    r"""TSD, for Time Shift Duration, is similar to MIDI-Like :ref:`MIDI-Like`
    but uses explicit *Duration* tokens to represent note durations, which have
    showed `better results than with *NoteOff* tokens <https://arxiv.org/abs/2002.00212>`_.
    If you specify ``use_programs`` as ``True`` in the config file, the tokenizer will
    add *Program* tokens before each *Pitch* tokens to specify its instrument, and will
    treat all tracks as a single stream of tokens.

    **Note:** as ``TSD`` uses *TimeShifts* events to move the time from note to note,
    it can be unsuited for tracks with pauses longer than the maximum *TimeShift*
    value. In such cases, the maximum *TimeShift* value will be used.
    **Note:** When decoding multiple token sequences (of multiple tracks), i.e. when
    ``config.use_programs`` is False, only the tempos and time signatures of the first
    sequence will be decoded for the whole MIDI.
    """

    def _tweak_config_before_creating_voc(self):
        if self.config.use_programs:
            self.one_token_stream = True

    def _add_time_events(self, events: List[Event], time_division: int) -> List[Event]:
        r"""Internal method intended to be implemented by inheriting classes.
        It creates the time events from the list of global and track events, and as
        such the final token sequence.

        :param events: note events to complete.
        :param time_division: time division of the MIDI being parsed.
        :return: the same events, with time events inserted.
        """
        # Add time events
        all_events = []
        previous_tick = 0
        previous_note_end = 0
        for event in events:
            # No time shift
            if event.time != previous_tick:
                # (Rest)
                if (
                    self.config.use_rests
                    and event.time - previous_note_end >= self._min_rest(time_division)
                ):
                    previous_tick = previous_note_end
                    rest_values = self._ticks_to_duration_tokens(
                        event.time - previous_tick, time_division, rest=True
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
                        *self._ticks_to_duration_tokens(time_shift, time_division)
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
            if event.type in ["Pitch", "PitchIntervalTime", "PitchIntervalChord"]:
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type in [
                "Program",
                "Tempo",
                "TimeSig",
                "Pedal",
                "PedalOff",
                "PitchBend",
                "Chord",
            ]:
                previous_note_end = max(previous_note_end, event.time)

        return all_events

    def _tokens_to_midi(
        self,
        tokens: Union[
            Union[TokSequence, List, np.ndarray, Any],
            List[Union[TokSequence, List, np.ndarray, Any]],
        ],
        programs: Optional[List[Tuple[int, bool]]] = None,
        time_division: int = TIME_DIVISION,
    ) -> Score:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence`,
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the
            MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.one_token_stream:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        midi = Score(time_division)
        if time_division % max(self.config.beat_res.values()) != 0:
            raise ValueError(
                f"Invalid time division, please give one divisible by"
                f"{max(self.config.beat_res.values())}"
            )

        # RESULTS
        tracks: Dict[int, Track] = {}
        tempo_changes, time_signature_changes = [], []

        def check_inst(prog: int):
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        current_instrument = None  # only use in when one_token_stream is False
        for si, seq in enumerate(tokens):
            # Set tracking variables
            current_tick = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = {prog: -128 for prog in self.config.programs}
            previous_pitch_chord = {prog: -128 for prog in self.config.programs}
            active_pedals = {}

            # Set track / sequence program if needed
            if not self.one_token_stream:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                current_instrument = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )

            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")
                if tok_type in ["TimeShift", "Rest"]:
                    if tok_type == "Rest":
                        current_tick = max(previous_note_end, current_tick)
                    current_tick += self._token_duration_to_ticks(
                        tok_val, time_division
                    )
                elif tok_type in ["Pitch", "PitchIntervalTime", "PitchIntervalChord"]:
                    if tok_type == "Pitch":
                        pitch = int(tok_val)
                        previous_pitch_onset[current_program] = pitch
                        previous_pitch_chord[current_program] = pitch
                    # We update previous_pitch_onset and previous_pitch_chord even if
                    # the try fails.
                    elif tok_type == "PitchIntervalTime":
                        pitch = previous_pitch_onset[current_program] + int(tok_val)
                        previous_pitch_onset[current_program] = pitch
                        previous_pitch_chord[current_program] = pitch
                    else:  # PitchIntervalChord
                        pitch = previous_pitch_chord[current_program] + int(tok_val)
                        previous_pitch_chord[current_program] = pitch
                    try:
                        vel_type, vel = seq[ti + 1].split("_")
                        dur_type, dur = seq[ti + 2].split("_")
                        if vel_type == "Velocity" and dur_type == "Duration":
                            dur = self._token_duration_to_ticks(dur, time_division)
                            new_note = Note(current_tick, dur, pitch, int(vel))
                            if self.one_token_stream:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_instrument.notes.append(new_note)
                            previous_note_end = max(
                                previous_note_end, current_tick + dur
                            )
                    except IndexError:
                        # A well constituted sequence should not raise an exception
                        # However with generated sequences this can happen, or if the
                        # sequence isn't finished
                        pass
                elif tok_type == "Program":
                    current_program = int(tok_val)
                elif tok_type == "Tempo" and si == 0:
                    tempo_changes.append(Tempo(current_tick, float(tok_val)))
                elif si == 0 and tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    time_signature_changes.append(TimeSignature(current_tick, num, den))
                elif tok_type == "Pedal":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if self.config.sustain_pedal_duration and ti + 1 < len(seq):
                        if seq[ti + 1].split("_")[0] == "Duration":
                            duration = self._token_duration_to_ticks(
                                seq[ti + 1].split("_")[1], time_division
                            )
                            # Add instrument if it doesn't exist, can happen for the
                            # first tokens
                            new_pedal = Pedal(current_tick, duration)
                            if self.one_token_stream:
                                check_inst(pedal_prog)
                                tracks[pedal_prog].pedals.append(new_pedal)
                            else:
                                current_instrument.pedals.append(new_pedal)
                    else:
                        if pedal_prog not in active_pedals:
                            active_pedals[pedal_prog] = current_tick
                elif tok_type == "PedalOff":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if pedal_prog in active_pedals:
                        new_pedal = Pedal(
                            active_pedals[pedal_prog],
                            current_tick - active_pedals[pedal_prog],
                        )
                        if self.one_token_stream:
                            check_inst(pedal_prog)
                            tracks[pedal_prog].pedals.append(
                                Pedal(
                                    active_pedals[pedal_prog],
                                    current_tick - active_pedals[pedal_prog],
                                )
                            )
                        else:
                            current_instrument.pedals.append(new_pedal)
                        del active_pedals[pedal_prog]
                elif tok_type == "PitchBend":
                    new_pitch_bend = PitchBend(current_tick, int(tok_val))
                    if self.one_token_stream:
                        check_inst(current_program)
                        tracks[current_program].pitch_bends.append(new_pitch_bend)
                    else:
                        current_instrument.pitch_bends.append(new_pitch_bend)

                if tok_type in [
                    "Program",
                    "Tempo",
                    "TimeSig",
                    "Pedal",
                    "PedalOff",
                    "PitchBend",
                    "Chord",
                ]:
                    previous_note_end = max(previous_note_end, current_tick)

            # Add current_inst to midi and handle notes still active
            if not self.one_token_stream:
                midi.tracks.append(current_instrument)

        # create MidiFile
        if self.one_token_stream:
            midi.tracks = list(tracks.values())
        midi.tempos = tempo_changes
        midi.time_signatures = time_signature_changes

        return midi

    def _create_base_vocabulary(self) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an
        underscore. Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MIDITokenizer`.

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
            first_note_token_type = (
                "Pitch" if self.config.program_changes else "Program"
            )
            dic["Program"] = ["Pitch"]
        else:
            first_note_token_type = "Pitch"  # noqa: S105
        dic["Pitch"] = ["Velocity"]
        dic["Velocity"] = ["Duration"]
        dic["Duration"] = [first_note_token_type, "TimeShift"]
        dic["TimeShift"] = [first_note_token_type, "TimeShift"]
        if self.config.use_pitch_intervals:
            for token_type in ("PitchIntervalTime", "PitchIntervalChord"):
                dic[token_type] = ["Velocity"]
                if self.config.use_programs:
                    dic["Program"].append(token_type)
                else:
                    dic["Duration"].append(token_type)
                    dic["TimeShift"].append(token_type)
        if self.config.program_changes:
            dic["Duration"].append("Program")

        if self.config.use_chords:
            dic["Chord"] = [first_note_token_type]
            dic["TimeShift"] += ["Chord"]
            if self.config.use_programs:
                dic["Program"].append("Chord")
            if self.config.use_pitch_intervals:
                dic["Chord"] += ["PitchIntervalTime", "PitchIntervalChord"]

        if self.config.use_tempos:
            dic["TimeShift"] += ["Tempo"]
            dic["Tempo"] = [first_note_token_type, "TimeShift"]
            if self.config.use_chords:
                dic["Tempo"] += ["Chord"]
            if self.config.use_rests:
                dic["Tempo"].append("Rest")  # only for first token
            if self.config.use_pitch_intervals:
                dic["Tempo"] += ["PitchIntervalTime", "PitchIntervalChord"]

        if self.config.use_time_signatures:
            dic["TimeShift"] += ["TimeSig"]
            dic["TimeSig"] = [first_note_token_type, "TimeShift"]
            if self.config.use_chords:
                dic["TimeSig"] += ["Chord"]
            if self.config.use_rests:
                dic["TimeSig"].append("Rest")  # only for first token
            if self.config.use_tempos:
                dic["TimeSig"].append("Tempo")
            if self.config.use_pitch_intervals:
                dic["TimeSig"] += ["PitchIntervalTime", "PitchIntervalChord"]

        if self.config.use_sustain_pedals:
            dic["TimeShift"].append("Pedal")
            if self.config.sustain_pedal_duration:
                dic["Pedal"] = ["Duration"]
                dic["Duration"].append("Pedal")
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
            if self.config.use_programs:
                dic["Program"].append("PitchBend")
            if not self.config.programs or self.config.program_changes:
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
            if self.config.use_pitch_intervals:
                dic["Rest"] += ["PitchIntervalTime", "PitchIntervalChord"]
        else:
            dic["TimeShift"].append("TimeShift")

        if self.config.program_changes:
            for token_type in [
                "TimeShift",
                "Rest",
                "PitchBend",
                "Pedal",
                "PedalOff",
                "Tempo",
                "TimeSig",
                "Chord",
            ]:
                if token_type in dic:
                    dic["Program"].append(token_type)
                    dic[token_type].append("Program")

        return dic
