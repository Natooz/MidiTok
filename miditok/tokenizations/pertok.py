"""PerTok tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from symusic import Note, Score, TimeSignature, Track

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import MIDI_INSTRUMENTS
from miditok.midi_tokenizer import MusicTokenizer

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

    Example usage:

    config = TokenizerConfig(
        pitch_range=(0, 127),
        num_velocities=32,
        use_pitchdrum_tokens=False,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        beat_grids=[(0, 8, 4), (0, 8, 6)],
        use_duration=True,
        use_velocity=True,
        use_microtiming=True,
        granularity = 480,
        max_microtiming_shift = 0.25,
        num_microtiming_bins = 32,
    )
    tokenizer = PerTok(config)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str or Path or None = None,
    ) -> None:
        super().__init__(tokenizer_config, params)
        self.test_vocab = self.vocab

    def _tweak_config_before_creating_voc(self) -> None:
        self.max_mt_shift = (
            self.config.additional_params["max_microtiming_shift"]
            * self.config.additional_params["granularity"]
        )

        self.use_duration = self.config.additional_params["use_duration"]
        self.use_velocity = self.config.additional_params["use_velocity"]
        self.use_microtiming = self.config.additional_params["use_microtiming"]

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
        self.tpq = self.config.additional_params["granularity"]
        tick_values = [0]

        for value in self.durations:
            beat, subdiv, resolution = value

            tick_value = int((beat + (subdiv / resolution)) * self.tpq)
            tick_values.append(tick_value)

        return np.array(sorted(set(tick_values)))

    def _create_durations_tuples(self) -> list[tuple[int, int, int]]:
        durations = set()

        for beat_grid in self.config.additional_params["beat_grids"]:
            start, end, resolution = beat_grid
            for beat in range(start, end):
                for subdiv in range(resolution):
                    if not (beat == 0 and subdiv == 0):
                        durations.add((beat, subdiv, resolution))
        return list(durations)

    def preprocess_score(self, score: Score) -> Score:
        """
        Change the symusic score resolution to the tokenizer's granularity.

        Args:
        ----
            score (Score): Symusic score

        Returns:
        -------
            Score: Symusic score that has the correct resolution

        """
        # Manually set our score granularity here, e.g. 480 ticks per quarter note
        new_tpq = self.config.additional_params["granularity"]
        if score.ticks_per_quarter != new_tpq:
            score = score.resample(new_tpq, min_dur=1)

        ticks_per_beat = None
        tpq_resampling_factors = None

        # Preprocess track events
        for t in range(len(score.tracks) - 1, -1, -1):
            if len(score.tracks[t].notes) == 0:
                del score.tracks[t]
                continue
            # Preprocesses notes
            score.tracks[t].notes = self._preprocess_notes(
                score.tracks[t].notes,
                score.tracks[t].is_drum,
                tpq_resampling_factors,
                ticks_per_beat,
            )

            if len(score.tracks[t].notes) == 0:
                del score.tracks[t]

        # A bit hacky but, if not included, create one of 4/4
        if len(score.time_signatures) == 0:
            score.time_signatures.append(
                TimeSignature(time=0, numerator=4, denominator=4)
            )

        return score

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

    def _create_track_events(self, track: Track, _: None = None) -> list[Event]:
        r"""
        Extract the tokens/events from a track (``symusic.Track``).

        Concerned events are: *Pitch*, *Velocity*, *Duration*, *NoteOn*, *NoteOff* and
        optionally *Chord*, *Pedal* and *PitchBend*.
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in** ``preprocess_score``.

        :param track: ``symusic.Track`` to extract events from.
        :param _: in place of ``ticks_per_beat``, unused here as Structured do not
            support time signatures, hence the ticks_per_beat value is always the same
            and equal to the score's time division.
        :return: sequence of corresponding ``Event``s.
        """
        # Make sure the notes are sorted first by their onset (start) times, second by
        # pitch: notes.sort(key=lambda x: (x.start, x.pitch)) done in score_to_tokens
        events = []

        # Creates the Note On, Note Off and Velocity events
        global_time = 0
        n_bars = -1
        add_ts_tokens = False

        # TODO: Right now hardcoded for 4/4 ; how to change this?
        ticks_per_bar = self.tpq * 4

        for note in track.notes:
            # Check if next notes time is in another bar;
            # if so, add 'bar' tokens and keep advancing clock
            while note.time // ticks_per_bar > n_bars:
                n_bars += 1
                global_time = n_bars * ticks_per_bar
                events.append(
                    Event(
                        type_="BAR",
                        time=global_time,
                        desc=f"{global_time} bar",
                        value="None",
                    )
                )

            time_delta = note.start - global_time
            closest_timeshift = int(
                self.get_closest_array_value(
                    value=time_delta, array=self.timeshift_tick_values
                )
            )

            # Timeshift
            if closest_timeshift != 0:
                events.append(
                    Event(
                        type_="TimeShift",
                        time=note.start,
                        desc=f"{closest_timeshift} timeshift",
                        value=closest_timeshift,
                    )
                )
                add_ts_tokens = True

            # Pitch
            pitch_token_name = (
                "PitchDrum"
                if track.is_drum and self.config.use_pitchdrum_tokens
                else "Pitch"
            )
            events.append(
                Event(
                    type_=pitch_token_name,
                    value=note.pitch,
                    time=note.start,
                    desc=note.end,
                )
            )

            if self.use_duration:
                closest_duration = self.get_closest_array_value(
                    value=note.duration, array=self.timeshift_tick_values
                )
                events.append(
                    Event(
                        type_="Duration",
                        value=closest_duration,
                        time=note.start,
                        desc=f"duration {note.duration}",
                    )
                )

            # Velocity
            if self.use_velocity:
                events.append(
                    Event(
                        type_="Velocity",
                        value=note.velocity,
                        time=note.start,
                        desc=f"{note.velocity}",
                    )
                )

            # Microtiming
            if self.use_microtiming:
                microtiming = time_delta - closest_timeshift
                closest_microtiming = int(
                    self.get_closest_array_value(
                        value=microtiming, array=self.microtiming_tick_values
                    )
                )
                events.append(
                    Event(
                        type_="MicroTiming",
                        value=closest_microtiming,
                        time=note.start,
                        desc=f"{closest_microtiming} microtiming shift",
                    )
                )

            if add_ts_tokens:
                global_time += closest_timeshift
                add_ts_tokens = False

        # Add a 'Bar' token at the end
        events.append(
            Event(
                type_="BAR", time=global_time, desc=f"{global_time} bar", value="None"
            )
        )

        return events

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence | list[TokSequence]:
        # Convert a **preprocessed** score object to a sequence of tokens.
        # Convert each track to tokens
        all_events = []
        if attribute_controls_indexes is None:
            attribute_controls_indexes = {}

        # Adds note tokens
        if not self.one_token_stream and len(score.tracks) == 0:
            all_events.append([])
        for track in score.tracks:
            note_events = self._create_track_events(track)
            if self.one_token_stream:
                all_events += note_events
            else:
                all_events.append(note_events)

        if self.one_token_stream:
            tok_sequence = TokSequence(events=all_events)
            self.complete_sequence(tok_sequence)
        else:
            tok_sequence = []
            for i in range(len(all_events)):
                tok_sequence.append(TokSequence(events=all_events[i]))
                self.complete_sequence(tok_sequence[-1])

        return tok_sequence

    def _tokens_to_score(
        self,
        tokens: TokSequence or list[TokSequence],
        programs: list[tuple[int, bool]] or None = None,
    ) -> Score:
        # Unsqueeze tokens in case of one_token_stream
        if self.one_token_stream:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens
        score = Score(self.tpq)

        # RESULTS
        instruments: dict[int, Track] = {}

        def check_inst(prog: int) -> None:
            if prog not in instruments:
                instruments[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        current_program = 0
        global_time = 0
        n_bars = -1
        ticks_per_bar = self.tpq * 4

        for si, seq in enumerate(tokens):
            # Set track / sequence program if needed
            if not self.one_token_stream:
                global_time = 0
                n_bars = -1
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                current_instrument = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name=(
                        "Drums"
                        if current_program == -1
                        else MIDI_INSTRUMENTS[current_program]["name"]
                    ),
                )

            for ti, token in enumerate(seq):
                token_type, token_val = token.split("_")
                # Advance clock forward to beginning of next bar
                if token_type == "BAR":
                    n_bars += 1
                    global_time += (ticks_per_bar * n_bars) - global_time

                elif token_type == "TimeShift":
                    global_time += int(token_val)

                elif token_type in ["Pitch", "PitchDrum"]:
                    try:
                        counter = ti
                        pitch = int(seq[counter].split("_")[1])

                        # Duration
                        if self.use_duration and (
                            seq[counter + 1].split("_")[0] == "Duration"
                        ):
                            counter += 1
                            duration = int(seq[counter].split("_")[1])
                        else:
                            duration = int(self.tpq)

                        # Velocity
                        if (
                            self.use_velocity
                            and seq[counter + 1].split("_")[0] == "Velocity"
                        ):
                            counter += 1
                            velocity = int(seq[counter].split("_")[1])
                        else:
                            velocity = 100

                        # Microtiming
                        if (
                            self.use_microtiming
                            and seq[counter + 1].split("_")[0] == "MicroTiming"
                        ):
                            counter += 1
                            microtiming = int(seq[counter].split("_")[1])
                        else:
                            microtiming = 0

                        time = max(int(global_time + microtiming), 0)
                        new_note = Note(time, duration, pitch, velocity)
                        if self.one_token_stream:
                            check_inst(current_program)
                            instruments[current_program].notes.append(new_note)
                        else:
                            current_instrument.notes.append(new_note)

                    except IndexError:
                        # A well constituted sequence should not raise an exception,
                        # however with generated sequences this can happen, or if the
                        # sequence isn't finished
                        pass

                elif token_type == "Program":
                    current_program = int(token_val)

            # Add current_inst to score and handle notes still active
            if not self.one_token_stream:
                score.tracks.append(current_instrument)

        # create scoreFile
        if self.one_token_stream:
            score.tracks = list(instruments.values())

        return score

    def _create_base_vocabulary(self) -> list[str]:
        vocab = ["BAR_None"]

        # NoteOn/NoteOff/Velocity
        self.timeshift_tick_values = self.create_timeshift_tick_values()
        self._add_note_tokens_to_vocab_list(vocab)

        # Timeshift
        vocab += [
            f"TimeShift_{timeshift!s}" for timeshift in self.timeshift_tick_values
        ]

        if self.use_duration:
            vocab += [
                f"Duration_{timeshift!s}" for timeshift in self.timeshift_tick_values
            ]

        # Microtiming
        if self.use_microtiming:
            mt_bins = self.config.additional_params["num_microtiming_bins"]
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

    def _add_time_events(self, events: list[Event], time_division: int) -> None:
        """
        Do not use this method.

        Args:
        ----
            events (list[Event]): None
            time_division (int): None

        """
