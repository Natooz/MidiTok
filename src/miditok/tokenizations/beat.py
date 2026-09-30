"""Beat-wise Encoding for Autoregressive Transformers (BEAT)."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from symusic import KeySignature, Note, Score, Tempo, TimeSignature, Track

from miditok.classes import Event, TokSequence
from miditok.constants import DEFAULT_VELOCITY, MIDI_INSTRUMENTS
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import (
    compute_ticks_per_bar,
    compute_ticks_per_beat,
    fix_offsets_overlapping_notes,
    get_bars_ticks,
    get_score_ticks_per_beat,
)
from miditok.utils.utils import np_get_closest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class BEAT(MusicTokenizer):
    r"""
    Encode a sparse three-state piano roll beat by beat.

    Introduced in `BEAT (Qian et al.) <https://openreview.net/forum?id=XrrGXLksji>`_.
    Each beat contains four steps (``beat_res`` is fixed). Time signatures are
    always enabled, and the denominator defines the beat unit: 6/8 has six
    eighth-note beats per bar. Missing time signatures default to 4/4.
    For each active pitch, a *Pattern* token encodes silence (0), onset (1), and
    sustain (2) as four base-3 digits, most significant digit first. Tracks are
    ordered by program within each beat, with drums last. Pitches descend within
    each track: the first *Pitch* is absolute, subsequent values are downward
    intervals. *PitchDrum* values, when enabled, remain absolute.
    All tracks share one token stream, with *Program* prefixes identifying their
    content within every beat, independently of ``program_changes``. Tracks with
    the same program stay separate and retain their input order. They each emit
    a *Program* block in every beat, using *Rest_None* when silent, so that their
    occurrence order identifies them across beats.

    *Bar* and *Beat* delimit the grid; empty beats contain *Rest_None*. Notes
    spanning beats use sustain patterns. Velocity is averaged over active steps
    for each pitch/beat and quantized to ``num_velocities`` bins. Decoded notes use
    the velocity of their onset beat. Overlapping notes of the same pitch are
    truncated at the next onset. These operations can lose information.

    Tempos, time signatures and optional key signatures are placed at bar
    boundaries. Key changes mapping to the same bar retain the last change.
    Chords, pedals, CCs, pitch bends and attribute controls are not supported.
    Durations, beat rests and relative pitches are intrinsic to this representation,
    independently of their optional flags in other tokenizers.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.TokenizerConfig` object.
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful if
        the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def _tweak_config_before_creating_voc(self) -> None:
        self.config.use_programs = True
        self.config.use_time_signatures = True
        self.config.one_token_stream_for_programs = True
        self._merge_same_program_tracks = False
        self.config.beat_res = {(0, 1): 4}
        self.config.use_note_duration_programs = self.config.programs
        self.config.use_rests = False  # Rest_None is structural, not a duration.
        self.config.use_pitch_intervals = False  # Pitch itself carries the interval.
        self.config.use_chords = False
        self.config.use_sustain_pedals = False
        self.config.sustain_pedal_duration = False
        self.config.use_control_changes = False
        self.config.use_pitch_bends = False
        self._disable_attribute_controls()

    def _preprocess_global_events(
        self, score: Score, resampling_factors: np.ndarray = None
    ) -> None:
        """
        Delay tempo/key changes to bar boundaries before quantization and deduplication.

        :param score: resampled score with preprocessed time signatures.
        :param resampling_factors: unused; BEAT places globals on bars, not steps.
        """
        del resampling_factors
        events = []
        if self.config.use_tempos:
            events.extend(score.tempos)
        if self.config.use_key_signatures:
            events.extend(score.key_signatures)
        if events:
            bar_ticks = get_bars_ticks(score)
            last_time_sig = next(
                ts for ts in reversed(score.time_signatures) if ts.time <= bar_ticks[-1]
            )
            # Include the next bar for changes occurring after the last note.
            bar_ticks.append(
                bar_ticks[-1]
                + compute_ticks_per_bar(last_time_sig, score.ticks_per_quarter)
            )
            for event in events:
                event.time = bar_ticks[np.searchsorted(bar_ticks, event.time)]
        super()._preprocess_global_events(score)

    def _preprocess_notes(
        self,
        track: Track,
        resampling_factors: np.ndarray = None,
        ticks_per_beat: np.ndarray = None,
        min_duration: int = 1,
    ) -> None:
        """
        Align note endpoints to the four-step grid and truncate same-pitch overlaps.

        :param track: track to preprocess inplace.
        :param resampling_factors: section end ticks and ticks per step, when the
            score contains different time-signature denominators.
        :param ticks_per_beat: unused; patterns do not use duration bins.
        :param min_duration: minimum note length in steps after quantization.
        """
        del ticks_per_beat
        if resampling_factors is not None:
            note_soa = track.notes.numpy()
            self._adjust_onsets_offsets(note_soa, resampling_factors, min_duration)
            track.notes = Note.from_numpy(**note_soa)

        # Reuse pitch filtering, velocity quantization and duplicate removal without
        # applying duration bins: a pattern can sustain a note for any number of beats.
        super()._preprocess_notes(track, min_duration=min_duration)

        # The piano roll holds only one note per pitch. A new onset ends the old note;
        # simultaneous onsets can leave zero-duration notes, which must be removed.
        track.notes.sort(key=lambda note: (note.time, note.pitch, note.duration))
        fix_offsets_overlapping_notes(track.notes)
        track.notes = [note for note in track.notes if note.duration > 0]

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: Mapping[int, Mapping[int, Sequence[int] | bool]]
        | None = None,
    ) -> TokSequence:
        """
        Assemble pitch patterns into beats, with tracks interleaved within each beat.

        :param score: preprocessed score with four steps per denominator-defined beat.
        :param attribute_controls_indexes: unused; attribute controls are unsupported.
        :return: one completed token sequence containing all tracks.
        """
        del attribute_controls_indexes
        global_events = self._create_global_events(score)
        globals_by_time: dict[int, list[Event]] = {}
        for event in global_events:
            globals_by_time.setdefault(event.time, []).append(event)
        end_tick = max([score.end(), *(event.time + 1 for event in global_events), 1])
        bars = get_bars_ticks(score)
        while bars[-1] < end_tick:
            time_sig = next(
                ts for ts in reversed(score.time_signatures) if ts.time <= bars[-1]
            )
            bars.append(
                bars[-1] + compute_ticks_per_bar(time_sig, score.ticks_per_quarter)
            )
        ticks_per_beat_sections = get_score_ticks_per_beat(score)

        tracks = sorted(
            score.tracks, key=lambda track: 128 if track.is_drum else track.program
        )
        num_tracks_per_program = Counter(
            -1 if track.is_drum else track.program for track in tracks
        )
        rolls = [
            track.pianoroll(["frame", "onset"], encode_velocity=True)
            for track in tracks
        ]
        events = []
        for bar_start, bar_end in pairwise(bars):
            ticks_per_beat = ticks_per_beat_sections[
                np.searchsorted(ticks_per_beat_sections[:, 0], bar_start, side="right"),
                1,
            ]
            events.append(Event("Bar", "None", bar_start))
            events.extend(globals_by_time.get(bar_start, []))
            for beat_start in range(bar_start, min(bar_end, end_tick), ticks_per_beat):
                events.append(Event("Beat", "None", beat_start))
                for track, roll in zip(tracks, rolls, strict=True):
                    segment = roll[
                        :,
                        :,
                        beat_start : beat_start + ticks_per_beat : ticks_per_beat // 4,
                    ]
                    pitches = np.flatnonzero(np.any(segment[0] > 0, axis=1))[::-1]
                    program = -1 if track.is_drum else track.program
                    # Same-program tracks need silent blocks to keep their positions.
                    if len(pitches) == 0 and num_tracks_per_program[program] == 1:
                        continue
                    events.append(Event("Program", program, beat_start))
                    if len(pitches) == 0:
                        events.append(Event("Rest", "None", beat_start))
                        continue
                    previous_pitch = None
                    for pitch in pitches:
                        frames, onsets = segment[:, pitch]
                        states = np.where(onsets > 0, 1, np.where(frames > 0, 2, 0))
                        pattern = int(states @ np.array([27, 9, 3, 1])[: len(states)])
                        is_drum = track.is_drum and self.config.use_pitchdrum_tokens
                        pitch_value = (
                            pitch
                            if previous_pitch is None or is_drum
                            else previous_pitch - pitch
                        )
                        events.append(
                            Event(
                                "PitchDrum" if is_drum else "Pitch",
                                int(pitch_value),
                                beat_start,
                            )
                        )
                        events.append(Event("Pattern", pattern, beat_start))
                        if self.config.use_velocities:
                            velocity = int(
                                np_get_closest(
                                    self.velocities,
                                    np.array([frames[frames > 0].mean()]),
                                )[0]
                            )
                            events.append(Event("Velocity", velocity, beat_start))
                        previous_pitch = pitch
                if events[-1].type_ == "Beat":
                    events.append(Event("Rest", "None", beat_start))
        sequence = TokSequence(
            events=events,
            _ticks_bars=[event.time for event in events if event.type_ == "Bar"],
            _ticks_beats=[event.time for event in events if event.type_ == "Beat"],
        )
        self.complete_sequence(sequence)
        return sequence

    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        """Time events are already assembled by ``_score_to_tokens``."""
        del time_division
        return events

    def _tokens_to_score(
        self,
        tokens: TokSequence,
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        """
        Reconstruct notes from onset and sustain patterns, including across beats.

        Malformed pitch/pattern groups and sustain states without an onset are
        ignored. Repeated programs identify separate tracks by occurrence order
        within each beat, including silent blocks.

        :param tokens: completed sequence to decode.
        :param programs: unused; programs are encoded in the sequence.
        :return: decoded score at this tokenizer's time division.
        """
        del programs
        score = Score(self.time_division)
        ticks_per_beat = self.time_division
        current_track = (0, 0)
        program_occurrences: Counter[int] = Counter()
        notes: dict[tuple[int, int], list[Note]] = {}
        active_notes: dict[tuple[tuple[int, int], int], Note] = {}
        previous_pitch = None
        bar_start = -1
        ticks_per_bar = 4 * self.time_division
        current_tick = 0
        for token_idx, token in enumerate(tokens.tokens):
            token_type, value = token.split("_")
            if token_type == "Bar":
                bar_start = 0 if bar_start < 0 else bar_start + ticks_per_bar
                current_tick = bar_start - ticks_per_beat
                previous_pitch = None
            elif token_type == "Beat":
                current_tick += ticks_per_beat
                previous_pitch = None
                program_occurrences.clear()
            elif token_type == "TimeSig":
                numerator, denominator = map(int, value.split("/"))
                time_sig = TimeSignature(max(bar_start, 0), numerator, denominator)
                ticks_per_beat = compute_ticks_per_beat(denominator, self.time_division)
                ticks_per_bar = numerator * ticks_per_beat
                current_tick = bar_start - ticks_per_beat
                score.time_signatures.append(time_sig)
            elif token_type == "Tempo":
                score.tempos.append(Tempo(max(bar_start, 0), float(value)))
            elif token_type == "KeySig":
                key, tonality = map(int, value.split(":"))
                score.key_signatures.append(
                    KeySignature(max(bar_start, 0), key, tonality)
                )
            elif token_type == "Program":
                program = int(value)
                current_track = (program, program_occurrences[program])
                program_occurrences[program] += 1
                notes.setdefault(current_track, [])
                previous_pitch = None
            elif token_type in {"Pitch", "PitchDrum"}:
                pitch_code = int(value)
                pitch = (
                    pitch_code
                    if previous_pitch is None or token_type == "PitchDrum"
                    else previous_pitch - pitch_code
                )
                previous_pitch = pitch
                group = tokens.tokens[token_idx + 1 : token_idx + 3]
                if (
                    not 0 <= pitch <= 127
                    or not group
                    or not group[0].startswith("Pattern_")
                ):
                    continue
                pattern = int(group[0].split("_")[1])
                if not 0 <= pattern < 81:
                    continue
                velocity = DEFAULT_VELOCITY
                if self.config.use_velocities:
                    if len(group) < 2 or not group[1].startswith("Velocity_"):
                        continue
                    velocity = int(group[1].split("_")[1])
                note_key = (current_track, pitch)
                step_ticks = ticks_per_beat // 4
                for step_idx, divisor in enumerate((27, 9, 3, 1)):
                    tick = current_tick + step_idx * step_ticks
                    state = pattern // divisor % 3
                    if state == 1:
                        note = Note(tick, step_ticks, pitch, velocity)
                        notes.setdefault(current_track, []).append(note)
                        active_notes[note_key] = note
                    elif (
                        state == 2
                        and note_key in active_notes
                        and active_notes[note_key].end == tick
                    ):
                        active_notes[note_key].duration += step_ticks
                    else:
                        active_notes.pop(note_key, None)
        for (program, _), track_notes in notes.items():
            score.tracks.append(
                Track(
                    program=0 if program == -1 else program,
                    is_drum=program == -1,
                    name="Drums"
                    if program == -1
                    else MIDI_INSTRUMENTS[program]["name"],
                    notes=sorted(
                        track_notes,
                        key=lambda note: (note.time, note.duration, note.pitch),
                    ),
                )
            )
        return score

    def _create_base_vocabulary(self) -> list[str]:
        """Create the pattern/grid vocabulary and reuse optional global tokens."""
        vocab = ["Bar_None", "Beat_None", "Rest_None"]
        vocab += [f"Pitch_{pitch}" for pitch in range(128)]
        vocab += [f"Pattern_{pattern}" for pattern in range(81)]
        if self.config.use_velocities:
            vocab += [f"Velocity_{velocity}" for velocity in self.velocities]
        self._add_additional_tokens_to_vocab_list(vocab)
        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        """Describe beat boundaries and pitch-pattern-velocity triples."""
        pitches = (
            {"Pitch", "PitchDrum"} if self.config.use_pitchdrum_tokens else {"Pitch"}
        )
        next_group = {"Beat", "Bar", "Program"} | pitches
        graph = {
            "Bar": {"Beat"},
            "Beat": {"Rest", "Program"},
            "Program": {"Rest"} | pitches,
            "Rest": {"Beat", "Bar", "Program"},
            "Pattern": {"Velocity"} if self.config.use_velocities else next_group,
        }
        for pitch_type in pitches:
            graph[pitch_type] = {"Pattern"}
        if self.config.use_velocities:
            graph["Velocity"] = next_group
        global_types = [
            token_type
            for enabled, token_type in (
                (self.config.use_time_signatures, "TimeSig"),
                (self.config.use_tempos, "Tempo"),
                (self.config.use_key_signatures, "KeySig"),
            )
            if enabled
        ]
        graph["Bar"].update(global_types)
        for index, token_type in enumerate(global_types):
            graph[token_type] = {"Beat", *global_types[index:]}
        return graph

    def _tokens_errors(self, tokens: list[str]) -> int:
        """Check token transitions and descending pitch codes within each track/beat."""
        errors = 0
        previous_type = None
        previous_pitch = None
        for token in tokens:
            token_type, value = token.split("_")
            invalid = (
                previous_type is not None
                and token_type not in self.tokens_types_graph[previous_type]
            )
            if token_type in {"Bar", "Beat", "Program"}:
                previous_pitch = None
            elif token_type in {"Pitch", "PitchDrum"}:
                pitch_code = int(value)
                pitch = (
                    pitch_code
                    if previous_pitch is None or token_type == "PitchDrum"
                    else previous_pitch - pitch_code
                )
                invalid |= not 0 <= pitch <= 127 or (
                    token_type == "Pitch"
                    and previous_pitch is not None
                    and pitch_code <= 0
                )
                previous_pitch = pitch
            errors += int(invalid)
            previous_type = token_type
        return errors
