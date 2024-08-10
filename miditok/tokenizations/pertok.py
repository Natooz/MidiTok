"""PerTok tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from symusic import Note, Score, TimeSignature, Track

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import MIDI_INSTRUMENTS
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import compute_ticks_per_beat
from miditok.constants import TIME_SIGNATURE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from numpy.typing import NDArray


class PerTok(MusicTokenizer):
    r"""
    PerTok: Performance Tokenizer.

    Designed to capture the full spectrum of rhythmic values (16ths, 32nds,
    various denominations of triplets/etc.) in addition to velocity
    and microtiming performance characteristics.
    It aims to achieve this while minimizing both vocabulary size and sequence length.
    Each note is characterized by 4-5 tokens:

    Timeshift (expressed in absolute ticks at score granularity*)
    Pitch
    Velocity
    Microtiming (also in score granularity)
    Optional: Duration (removed for drums to reduce seq. length)

    *'Timeshift' tokens are expressed as the quantized time value, as one would expect
    in a sheet music score.
    E.g. at 480 ticks per quarter note, a 16th note would be
    expressed as 'Timeshift_120'.
    The microtiming shift is then characterized as the subtle deviation from
    this quantized value.
    E.g. for a slightly delayed 1/8th note:
    'Timeshift_240', 'Pitch_60', 'Velocity_100', 'Microtiming_12'

    Additionally, 'Bar' tokens are inserted at the start of each new measure.
    This helps further reduceseq. length and
    theoretically reduces the 'timing drift' models can develop at longer seq. lengths.

    New TokenizerConfig Options:

    beat_grids: now allows multiple overlapping values
    use_duration: include duration tokens
    use_velocity: include velocity tokens
    use_microtiming: include microtiming tokens
    granularity: Granularity of rhythm as measured per quarter note,
    e.g. 480 ticks-per-quarter (tpq).
    max_microtiming_shift: Maximum timeshift in microtiming tokens
    num_microtiming_bins: Total number of microtiming tokens
    
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str or Path or None = None,
    ) -> None:
        super().__init__(tokenizer_config, params)
        self.test_vocab = self.vocab

    def _tweak_config_before_creating_voc(self) -> None:

        # Microtiming
        # TPQ value of maximum range of microtiming tokens
        self.use_microtiming = self.config.use_microtiming = True
        
        self.max_mt_shift = (
            self.config.max_microtiming_shift 
            * self.config.res_microtiming
        )
        
        self.use_velocity = self.config.use_velocities
        
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_tempos = False
        self.config.use_time_signatures = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.program_changes = False

    def create_timeshift_tick_values(self) -> NDArray:
        """
        Generate tick-based timeshift tokens.

        Returns
        -------
            NDArray: Array of available timeshift values

        """
        self.tpq = self.config.max_num_pos_per_beat
        tick_values = [0]

        for value in self.durations:
            beat, subdiv, resolution = value

            tick_value = int((beat + (subdiv / resolution)) * self.tpq)
            tick_values.append(tick_value)

        return np.array(sorted(set(tick_values)))

    def _create_durations_tuples(self) -> list[tuple[int, int, int]]:
        durations = []
        tpq = self.config.max_num_pos_per_beat
        
        for beat_range, resolution in self.config.beat_res.items():
            start, end = beat_range
            for beat in range(start, end):
                for subdiv in range(resolution):
                    if not (beat == 0 and subdiv == 0):
                        subres = (tpq//resolution * subdiv) if subdiv != 0 else 0
                        durations.append((beat, subres, tpq))

        # [120, 220, 330] -> timeshift of '25' should result in no timeshift (only microtime)
        self.min_timeshift = int(min([(beat*res + subres) for beat, subres, res in durations]) * 0.5)
        
        return durations

    def get_closest_array_value(
        self, value: int | float, array: NDArray
    ) -> int | float:
        """
        Find the closest value from a given numpy array.

        Args:
        ----
            value (float | int): The input value to query
            array (NDArray): Numpy array of possible values

        Returns:
        -------
            int | float: The closest value to the input

        """
        return array[np.abs(array - value).argmin()]

    def get_closest_duration_tuple(self, target):
        return min(self.durations, key=lambda x: abs(((x[0]*x[-1]+x[1]) - target)))


    def _create_base_vocabulary(self) -> list[str]:
        vocab = ["BAR_None"]

        # NoteOn/NoteOff/Velocity
        self.timeshift_tick_values = self.create_timeshift_tick_values()
        self._add_note_tokens_to_vocab_list(vocab)

        # TimeShift
        vocab += [
            f"TimeShift_{self.duration_tuple_to_str(duration)}" for duration in self.durations
        ]

        # Duration
        if any(self.config.use_note_duration_programs):
            vocab += [
                f"Duration_{self.duration_tuple_to_str(duration)}" for duration in self.durations
            ]
        
        # Microtiming
        if self.use_microtiming:
            mt_bins = self.config.num_microtiming_bins
            self.microtiming_tick_values = np.linspace(
                -self.max_mt_shift, self.max_mt_shift, mt_bins + 1, dtype=np.intc
            )

            vocab += [
                f"MicroTiming_{microtiming!s}"
                for microtiming in self.microtiming_tick_values
            ]

        self._add_additional_tokens_to_vocab_list(vocab)

        return vocab

    def _add_note_tokens_to_vocab_list(self, vocab: list[str]) -> None:
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        if self.use_velocity:
            vocab += [f"Velocity_{i}" for i in self.velocities]

    def _create_token_types_graph(self) -> dict[str, list[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic = {
            "Pitch": {"Velocity"},
            "PitchDrum": {"Velocity"},
            "Velocity": {"Duration"},
            "Duration": {"TimeShift"},
            "TimeShift": {"Pitch", "PitchDrum"},
        }

        if self.config.use_programs:
            dic["Program"] = {"Pitch", "PitchDrum"}
            dic["TimeShift"] = {"Program"}

        return dic
    
    def duration_tuple_to_str(self, duration_tuple):
       return ".".join(str(x) for x in duration_tuple)
    
    def _create_duration_event(
        self,
        note: Note,
        program,
        ticks_per_beat: np.ndarray,
        tpb_idx
    ) -> Event:
        
        
        duration_tuple = self.get_closest_duration_tuple(note.duration)
        duration = ".".join(str(x) for x in duration_tuple)
        
        return Event(
            type_="Duration",
            value=duration,
            time=note.start,
            program=program,
            desc=f"duration {note.duration}"
        )
                
    def _add_time_events(self, events: list[Event], time_division: int) -> None:


        # Add time events
        all_events = []
        previous_tick = 0
        ticks_per_beat = compute_ticks_per_beat(TIME_SIGNATURE[1], time_division)
        ticks_per_bar = ticks_per_beat * TIME_SIGNATURE[0]
        curr_bar = 0
  
        for event in events:
            
            # Bar
            bar_time = previous_tick
            while event.time > ((curr_bar+1) * ticks_per_bar - self.min_timeshift):
                bar_time += ticks_per_bar - (bar_time % ticks_per_bar) # tpq=220, time=20, so add 200 to get to next bar
                
                all_events.append(
                    Event(
                        type_="BAR",
                        value=None,
                        time=bar_time,
                        desc=f"Bar {bar_time}"
                    )
                )
                
                curr_bar += 1
                previous_tick = curr_bar * ticks_per_bar
            
            
            # Time Signature
            if event.type_ == "TimeSig":
                ticks_per_beat = compute_ticks_per_beat(
                    int(event.value.split("/")[1]), time_division
                )
                ticks_per_bar = ticks_per_beat * int(event.value.split("/")[0])


            time_delta = event.time - previous_tick
            timeshift = 0
            
            # Time Shift
            # Only should be placed before 'Pitch' events
            if time_delta >= self.min_timeshift and event.type_ == "Pitch":
                
                ts_tuple = self.get_closest_duration_tuple(time_delta)
                ts = ".".join(str(x) for x in ts_tuple)
                
                all_events.append(
                    Event(
                        type_="TimeShift",
                        value=ts,
                        time=event.time,
                        desc=f"timeshift {ts}",
                    )
                )
                timeshift = ts_tuple[0] * ts_tuple[-1] + ts_tuple[1]
                previous_tick += timeshift
            
            all_events.append(event)
            
            # Microtiming
            # Right now hard-coded to come only after 'Pitch' tokens
            if self.use_microtiming and event.type_ == "Pitch":
                microtiming = time_delta - timeshift
                closest_microtiming = int(
                    self.get_closest_array_value(
                        value=microtiming, array=self.microtiming_tick_values
                    )
                )
                all_events.append(
                    Event(
                        type_="MicroTiming",
                        value=closest_microtiming,
                        time=event.time,
                        desc=f"{closest_microtiming} microtiming",
                    )
                )

        return all_events
    
    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.config.one_token_stream_for_programs:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        score = Score(self.time_division)
        
        mt_offset = 1 if self.use_microtiming else 0
        vel_offset = (mt_offset + 1) if self.use_velocity else mt_offset
        dur_offset = vel_offset + 1
        
        
        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes, time_signature_changes = [], []

        def check_inst(prog: int) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        current_track = None  # used only when one_token_stream is False
        for si, seq in enumerate(tokens):
            # Set tracking variables
            current_tick = 0
            curr_bar = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = {prog: -128 for prog in self.config.programs}
            previous_pitch_chord = {prog: -128 for prog in self.config.programs}
            active_pedals = {}
            ticks_per_beat = score.ticks_per_quarter
            ticks_per_bar = ticks_per_beat * TIME_SIGNATURE[0]

            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                elif self.config.use_programs:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type.startswith("Program"):
                            current_program = int(tok_val)
                            if current_program == -1:
                                is_drum, current_program = True, 0
                            break
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )
            current_track_use_duration = (
                current_program in self.config.use_note_duration_programs
            )

            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")
                
                if tok_type == "BAR":
                    curr_bar += 1
                    current_tick += (ticks_per_bar * curr_bar) - current_tick
                elif tok_type == "TimeShift":
                    current_tick += self._tpb_tokens_to_ticks[ticks_per_beat][tok_val]
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                elif tok_type in [
                    "Pitch",
                    "PitchDrum",
                    "PitchIntervalTime",
                    "PitchIntervalChord",
                ]:
                    if tok_type in {"Pitch", "PitchDrum"}:
                        pitch = int(tok_val)
                    elif tok_type == "PitchIntervalTime":
                        pitch = previous_pitch_onset[current_program] + int(tok_val)
                    else:  # PitchIntervalChord
                        pitch = previous_pitch_chord[current_program] + int(tok_val)
                    if (
                        not self.config.pitch_range[0]
                        <= pitch
                        <= self.config.pitch_range[1]
                    ):
                        continue

                    # We update previous_pitch_onset and previous_pitch_chord even if
                    # the try fails.
                    if tok_type != "PitchIntervalChord":
                        previous_pitch_onset[current_program] = pitch
                    previous_pitch_chord[current_program] = pitch

                    try:
                        if self.use_microtiming:
                            mt_type, mt = seq[ti + mt_offset].split("_")
                            mt = int(mt)
                        else:
                            mt_type, mt = "MicroTiming", 0
                        if self.config.use_velocities:
                            vel_type, vel = seq[ti + vel_offset].split("_")
                        else:
                            vel_type, vel = "Velocity", DEFAULT_VELOCITY
                        if current_track_use_duration:
                            dur_type, dur = seq[ti + dur_offset].split("_")
                        else:
                            dur_type = "Duration"
                            dur = int(
                                self.config.default_note_duration * ticks_per_beat
                            )
                        if mt_type == "MicroTiming" and vel_type == "Velocity" and dur_type == "Duration":
                            if isinstance(dur, str):
                                dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                            mt += current_tick
                            new_note = Note(mt, dur, pitch, int(vel))
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
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
                    current_track_use_duration = (
                        current_program in self.config.use_note_duration_programs
                    )
                    if (
                        not self.config.one_token_stream_for_programs
                        and self.config.program_changes
                    ):
                        if current_program != -1:
                            current_track.program = current_program
                        else:
                            current_track.program = 0
                            current_track.is_drum = True
                elif tok_type == "Tempo" and si == 0:
                    tempo_changes.append(Tempo(current_tick, float(tok_val)))
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    if si == 0:
                        time_signature_changes.append(
                            TimeSignature(current_tick, num, den)
                        )
                    ticks_per_beat = self._tpb_per_ts[den]
                    ticks_per_bar = ticks_per_beat * num
                elif tok_type == "Pedal":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if self.config.sustain_pedal_duration and ti + 1 < len(seq):
                        if seq[ti + 1].split("_")[0] == "Duration":
                            duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                                seq[ti + 1].split("_")[1]
                            ]
                            # Add instrument if it doesn't exist, can happen for the
                            # first tokens
                            new_pedal = Pedal(current_tick, duration)
                            if self.config.one_token_stream_for_programs:
                                check_inst(pedal_prog)
                                tracks[pedal_prog].pedals.append(new_pedal)
                            else:
                                current_track.pedals.append(new_pedal)
                    elif pedal_prog not in active_pedals:
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
                        if self.config.one_token_stream_for_programs:
                            check_inst(pedal_prog)
                            tracks[pedal_prog].pedals.append(
                                Pedal(
                                    active_pedals[pedal_prog],
                                    current_tick - active_pedals[pedal_prog],
                                )
                            )
                        else:
                            current_track.pedals.append(new_pedal)
                        del active_pedals[pedal_prog]
                elif tok_type == "PitchBend":
                    new_pitch_bend = PitchBend(current_tick, int(tok_val))
                    if self.config.one_token_stream_for_programs:
                        check_inst(current_program)
                        tracks[current_program].pitch_bends.append(new_pitch_bend)
                    else:
                        current_track.pitch_bends.append(new_pitch_bend)

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

            # Add current_inst to the score and handle notes still active
            if not self.config.one_token_stream_for_programs and not is_track_empty(
                current_track
            ):
                score.tracks.append(current_track)

        # Add global events to the score
        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes

        return score