"""PerTok tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from symusic import Note, Pedal, PitchBend, Score, Tempo, TimeSignature, Track

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS, TIME_SIGNATURE
from miditok.midi_tokenizer import MusicTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from symusic.core import TimeSignatureTickList


class PerTok(MusicTokenizer):
    r"""
    PerTok: Performance Tokenizer.

    Created by Lemonaide
    https://www.lemonaide.ai/

    Designed to capture the full spectrum of rhythmic values
    (16ths, 32nds, various denominations of triplets/etc.)
    in addition to velocity and microtiming performance characteristics.
    It aims to achieve this while minimizing both vocabulary size and sequence length.

    Notes are encoded by 2-5 tokens:

    * TimeShift;
    * Pitch;
    * Velocity (optional);
    * MicroTiming (optional);
    * Duration (optional).

    *Timeshift* tokens are expressed as the nearest quantized value
    based upon *beat_res* parameters.
    The microtiming shift is then characterized as the remainder from
    this quantized value. Timeshift and MicroTiming are represented
    in the full ticks-per-quarter (tpq) resolution, e.g. 480 tpq.

    Additionally, *Bar* tokens are inserted at the start of each new measure.
    This helps further reduce seq. length and potentially reduces the timing drift
    models can develop at longer seq. lengths.

    New TokenizerConfig Options:

    * beat_res: now allows multiple, overlapping values;
    * ticks_per_quarter: resolution of the MIDI timing data;
    * use_microtiming: inclusion of MicroTiming tokens;
    * max_microtiming_shift: float value of the farthest distance of MicroTiming shifts;
    * num_microtiming_bins: total number of MicroTiming tokens.

    Example Tokenizer Config:

    .. code-block:: python

        TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 4, (0, 4): 3},
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": False,
        "use_rests": False,
        "use_tempos": False,
        "use_time_signatures": True,
        "use_programs": False,
        "use_microtiming": True,
        "ticks_per_quarter": 320,
        "max_microtiming_shift": 0.125,
        "num_microtiming_bins": 30,
        "use_position_toks": true
        }
        config = TokenizerConfig(**TOKENIZER_PARAMS)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str or Path or None = None,
    ) -> None:
        super().__init__(tokenizer_config, params)
        if "ticks_per_quarter" not in self.config.additional_params:
            msg = "Tokenizer config must have a value for ticks_per_quarter"
            raise ValueError(msg)

        # Events which will use a "MicroTiming" token
        self.microtime_events = [
            "Pitch",
            "Pedal",
            "PedalOff",
            "PitchIntervalChord",
            "PitchBend",
            "Chord",
            "PitchDrum",
            "Program",
        ]
        # This will be hit when we're using microtiming
        # and have loaded a TRAINED tokenizer
        if self.config.additional_params["use_microtiming"] and not hasattr(
            self, "microtiming_tick_values"
        ):
            self.microtiming_tick_values = self.create_microtiming_tick_values()

    def _tweak_config_before_creating_voc(self) -> None:
        self.tpq = self.config.additional_params["ticks_per_quarter"]
        self.use_microtiming = self.config.additional_params["use_microtiming"]
        if self.use_microtiming:
            mt_keys = ["max_microtiming_shift", "num_microtiming_bins"]
            if missing := set(mt_keys) - set(self.config.additional_params.keys()):
                msg = f"TokenizerConfig is missing required keys: {', '.join(missing)}"
                raise ValueError(msg)

        self.max_mt_shift = (
            self.config.additional_params["max_microtiming_shift"] * self.tpq
        )

        self.use_position_toks: bool = self.config.additional_params.get(
            "use_position_toks", False
        )
        self.position_locations = None

    def _create_base_vocabulary(self) -> list[str]:
        vocab = ["Bar_None"]

        # NoteOn/NoteOff/Velocity
        self.timeshift_tick_values = self.create_timeshift_tick_values()
        self._add_note_tokens_to_vocab_list(vocab)

        # Position tokens - for denoting absolute positions of tokens
        if self.use_position_toks:
            self.position_locations = self._create_position_tok_locations()
            vocab += [f"Position_{pos}" for pos in self.position_locations]

        # PerTok's original version uses 'Timeshift' to denote delta between
        # two note's positions
        else:
            vocab += [
                f"TimeShift_{self._duration_tuple_to_str(duration)}"
                for duration in self.durations
            ]

        # Duration
        if any(self.config.use_note_duration_programs):
            vocab += [
                f"Duration_{self._duration_tuple_to_str(duration)}"
                for duration in self.durations
            ]

        # Microtiming
        if self.config.additional_params["use_microtiming"]:
            self.microtiming_tick_values = self.create_microtiming_tick_values()
            vocab += [
                f"MicroTiming_{microtiming!s}"
                for microtiming in self.microtiming_tick_values
            ]

            # Add additional tokens
        self._add_additional_tokens_to_vocab_list(vocab)

        return list(dict.fromkeys(vocab))

    # Methods to override base MusicTokenizer versions
    # To handle MicroTiming and multiple beat_res resolutions
    # This is accomplished by removing the downsampling methods
    # As a result, many time-based methods need to be redesigned

    def _resample_score(
        self, score: Score, _new_tpq: int, _time_signatures_copy: TimeSignatureTickList
    ) -> Score:
        if score.ticks_per_quarter != self.tpq:
            score = score.resample(self.tpq, min_dur=1)

        return score

    def _adjust_durations(
        self, notes_pedals_soa: dict[str, np.ndarray], ticks_per_beat: np.ndarray
    ) -> None:
        pass

    def _create_duration_event(
        self, note: Note, _program: int, _ticks_per_beat: np.ndarray, _tpb_idx: int
    ) -> Event:
        duration_tuple = self._get_closest_duration_tuple(note.duration)
        duration = ".".join(str(x) for x in duration_tuple)

        return Event(
            type_="Duration",
            value=duration,
            time=note.start,
            program=_program,
            desc=f"duration {note.duration}",
        )

    def create_microtiming_tick_values(self) -> NDArray:
        """
        Generate tick-based microtiming tokens.

        Returns
        -------
            NDArray: Array of available microtiming values

        """
        mt_bins = self.config.additional_params["num_microtiming_bins"]
        return np.linspace(
            -self.max_mt_shift, self.max_mt_shift, mt_bins + 1, dtype=np.intc
        )

    def create_timeshift_tick_values(self) -> NDArray:
        """
        Generate tick-based timeshift tokens.

        Returns
        -------
            NDArray: Array of available timeshift values

        """
        tick_values = [0]

        for value in self.durations:
            beat, subdiv, resolution = value
            tick_value = int((beat + (subdiv / resolution)) * self.tpq)
            tick_values.append(tick_value)

        return np.array(sorted(set(tick_values)))

    def _create_durations_tuples(self) -> list[tuple[int, int, int]]:
        durations = []

        for beat_range, resolution in self.config.beat_res.items():
            start, end = beat_range
            for beat in range(start, end):
                for subdiv in range(resolution):
                    if not (beat == 0 and subdiv == 0):
                        subres = (self.tpq // resolution * subdiv) if subdiv != 0 else 0
                        durations.append((beat, subres, self.tpq))

        self.min_timeshift = int(
            min([(beat * res + subres) for beat, subres, res in durations]) * 0.5
        )

        return durations

    # Utility Methods
    @staticmethod
    def _get_closest_array_value(value: int | float, array: NDArray) -> int | float:
        return array[np.abs(array - value).argmin()]

    def _get_closest_duration_tuple(self, target: int) -> tuple[int, int, int]:
        return min(self.durations, key=lambda x: abs((x[0] * x[-1] + x[1]) - target))

    # Given a note's location, find the closest Position token
    def _find_closest_position_tok(self, target: int) -> int:
        return min(self.position_locations, key=lambda x: abs(x - target))

    @staticmethod
    def _convert_durations_to_ticks(duration: str) -> int:
        beats, subdiv, tpq = map(int, duration.split("."))
        return beats * tpq + subdiv

    def _create_position_tok_locations(self) -> list[int]:
        return [0, *sorted({(dur[0] * self.tpq) + dur[1] for dur in self.durations})]

    @staticmethod
    def _duration_tuple_to_str(duration_tuple: tuple[int, int, int]) -> str:
        return ".".join(str(x) for x in duration_tuple)

    def _add_time_events(self, events: list[Event], _time_division: int) -> list[Event]:
        # Add time events
        all_events = []
        previous_tick = 0
        ticks_per_bar = self.tpq * TIME_SIGNATURE[0]
        curr_bar = 0
        bar_time = 0  # to keep track of location of last bar start
        pos = 0
        position_in_bar = 0
        microtiming = 0

        for event in events:
            # Bar
            global_time = previous_tick
            while event.time > ((curr_bar + 1) * ticks_per_bar - self.min_timeshift):
                global_time += ticks_per_bar - (
                    global_time % ticks_per_bar
                )  # tpq=220, time=20, so add 200 to get to next bar

                bar_time += ticks_per_bar
                position_in_bar = 0

                all_events.append(
                    Event(
                        type_="Bar", value=None, time=bar_time, desc=f"Bar {bar_time}"
                    )
                )

                curr_bar += 1
                previous_tick = curr_bar * ticks_per_bar

            # Time Signature
            if event.type_ == "TimeSig":
                num, den = self._parse_token_time_signature(event.value)
                ticks_per_bar = den / 4 * num * self.tpq

            time_delta = event.time - previous_tick

            # Option 1: Adding Position tokens
            if event.type_ in self.microtime_events:
                if self.use_position_toks:
                    position_in_bar = event.time - bar_time
                    pos = self._find_closest_position_tok(position_in_bar)
                    microtiming = position_in_bar - pos
                    if time_delta >= self.min_timeshift:
                        all_events.append(
                            Event(
                                type_="Position",
                                value=pos,
                                time=event.time,
                                desc=f"position {pos}",
                            )
                        )
                        previous_tick = bar_time + pos

                # Option 2: creating Timeshift tokens
                else:
                    timeshift = 0
                    if time_delta >= self.min_timeshift:
                        ts_tuple = self._get_closest_duration_tuple(time_delta)
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
                    microtiming = time_delta - timeshift

            all_events.append(event)

            # MicroTiming
            # These will come only after certain types of tokens, e.g. 'Pitch'
            if self.use_microtiming and event.type_ in self.microtime_events:
                closest_microtiming = int(
                    self._get_closest_array_value(
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
        score = Score(self.tpq)

        mt_offset = 1 if self.use_microtiming else 0
        vel_offset = (mt_offset + 1) if self.config.use_velocities else mt_offset
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
        ticks_per_beat = score.ticks_per_quarter
        for si, seq in enumerate(tokens):
            # Set tracking variables
            current_tick = 0
            curr_bar = 0
            curr_bar_tick = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = dict.fromkeys(self.config.programs, -128)
            previous_pitch_chord = dict.fromkeys(self.config.programs, -128)
            active_pedals = {}
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

                if tok_type == "Bar":
                    curr_bar += 1
                    current_tick += (ticks_per_bar * curr_bar) - current_tick
                    curr_bar_tick = current_tick
                elif tok_type == "TimeShift":
                    current_tick += self._convert_durations_to_ticks(tok_val)
                elif tok_type == "Position":
                    current_tick = curr_bar_tick + int(tok_val)
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
                        if (
                            self.use_microtiming
                            and "MicroTiming" in seq[ti + mt_offset]
                        ):
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
                        if (
                            mt_type == "MicroTiming"
                            and vel_type == "Velocity"
                            and dur_type == "Duration"
                        ):
                            if isinstance(dur, str):
                                dur = self._convert_durations_to_ticks(dur)

                            mt += current_tick
                            new_note = Note(int(mt), dur, pitch, int(vel))
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
                            previous_note_end = max(previous_note_end, mt + dur)
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
                    ticks_per_bar = den / 4 * num * ticks_per_beat
                    if si == 0:
                        time_signature_changes.append(
                            TimeSignature(int(current_tick), num, den)
                        )

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
                            new_pedal = Pedal(int(current_tick), int(duration))
                            if self.config.one_token_stream_for_programs:
                                check_inst(pedal_prog)
                                tracks[pedal_prog].pedals.append(new_pedal)
                            else:
                                current_track.pedals.append(new_pedal)
                    elif pedal_prog not in active_pedals:
                        active_pedals[pedal_prog] = int(current_tick)
                elif tok_type == "PedalOff":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if pedal_prog in active_pedals:
                        new_pedal = Pedal(
                            int(active_pedals[pedal_prog]).int(
                                current_tick - active_pedals[pedal_prog]
                            ),
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
        if time_signature_changes is None:
            num, den = TIME_SIGNATURE
            time_signature_changes.append(TimeSignature(0, num, den))
        score.time_signatures = time_signature_changes

        return score

    def _tokens_errors(self, _tokens: list[str | list[str]]) -> int:
        return 0

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        # Bar, TimeSig, TimeShift, Pitch, MT, Velocity, Duration

        dic: dict[str, set[str]] = {}
        return dic
