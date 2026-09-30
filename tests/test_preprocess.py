"""Tests on the preprocessing steps of music files, before tokenization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from symusic import (
    ControlChange,
    KeySignature,
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

import miditok

from .utils_tests import MIDI_PATHS_ALL, adjust_tok_params_for_tests

if TYPE_CHECKING:
    from pathlib import Path

CONFIG_KWARGS = {
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "log_tempos": True,
    "beat_res": {(0, 4): 8, (4, 12): 4, (12, 16): 2},
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
}
TOKENIZATIONS = ["MIDILike", "TSD"]


@pytest.mark.parametrize(
    ("times", "expected", "sorted_times"),
    [
        (
            [0, 11, 12, 15, 16, 17, 75, 76, 83],
            [0, 11, 12, 12, 12, 20, 76, 76, 84],
            True,
        ),
        ([77, 83, 160], [76, 84, 160], True),
        ([83, 11, 17, 133, 76], [84, 11, 20, 132, 76], False),
        ([], [], True),
    ],
)
def test_time_quantization_sections(
    times: list[int], expected: list[int], sorted_times: bool
) -> None:
    """Handle exact boundaries, empty sections and unsorted timestamps consistently."""
    timestamps = np.array(times, dtype=np.int32)
    sections = np.array([[12, 1], [76, 8], [124, 4]])
    miditok.MusicTokenizer._adjust_time_to_tpb(
        timestamps, sections, sorted_times=sorted_times
    )
    assert timestamps.tolist() == expected


@pytest.mark.parametrize("tokenization", ["BEAT", "REMI", "TSD", "MIDILike"])
def test_global_changes_across_time_signatures(tokenization: str) -> None:
    """Quantize metadata on each meter's grid, or at bar boundaries for BEAT."""
    tokenizer = getattr(miditok, tokenization)(
        miditok.TokenizerConfig(
            beat_res={(0, 4): 4},
            use_tempos=True,
            use_key_signatures=True,
            use_time_signatures=True,
            time_signature_range={16: [3], 2: [2], 4: [3, 4]},
            tempo_range=(60, 180),
            num_tempos=3,
            delete_equal_successive_tempo_changes=False,
        )
    )
    score = Score(16)
    score.time_signatures = [
        TimeSignature(0, 3, 16),
        TimeSignature(12, 2, 2),
        TimeSignature(76, 3, 4),
    ]
    # Changes continue after the last note and collide on the quantized grid.
    times = [0, 15, 16, 17, 47, 49, 77, 78, 83]
    keys = [1, 2, 3, 4, 5, 6, -1, -2, -3]
    tempos = [60, 120, 180, 60, 120, 180, 60, 120, 180]
    score.key_signatures = [
        KeySignature(time, key, 0) for time, key in zip(times, keys, strict=True)
    ]
    score.tempos = [
        Tempo(time, tempo) for time, tempo in zip(times, tempos, strict=True)
    ]
    score.tracks = [Track(notes=[Note(0, 4, 60, 80)])]
    original = score.copy()

    if tokenization == "BEAT":
        expected_times = [0, 76, 124]
        expected_keys = [1, 6, -3]
        expected_tempos = [60, 180, 180]
    else:
        # The half-note grid begins at tick 12, so its steps are 12, 20, 28, ...
        expected_times = [0, 12, 20, 44, 52, 76, 84]
        expected_keys = [1, 3, 4, 5, 6, -2, -3]
        expected_tempos = [60, 180, 60, 120, 180, 120, 180]
    expected_key_changes = [
        KeySignature(time, key, 0)
        for time, key in zip(expected_times, expected_keys, strict=True)
    ]

    preprocessed = tokenizer.preprocess_score(score)
    assert score == original
    assert list(preprocessed.key_signatures) == expected_key_changes
    assert [(tempo.time, round(tempo.tempo)) for tempo in preprocessed.tempos] == list(
        zip(expected_times, expected_tempos, strict=True)
    )
    assert tokenizer.preprocess_score(preprocessed) == preprocessed
    decoded = tokenizer.decode(tokenizer.encode(preprocessed, no_preprocess_score=True))
    assert list(decoded.key_signatures) == expected_key_changes
    assert [(tempo.time, round(tempo.tempo)) for tempo in decoded.tempos] == list(
        zip(expected_times, expected_tempos, strict=True)
    )


@pytest.mark.parametrize("tokenization", ["REMI", "TSD", "MIDILike"])
def test_track_events_across_time_signatures(tokenization: str) -> None:
    """Use one time grid for notes, CCs, pitch bends and pedal on/off events."""
    tokenizer = getattr(miditok, tokenization)(
        miditok.TokenizerConfig(
            beat_res={(0, 4): 4},
            use_time_signatures=True,
            time_signature_range={16: [3], 2: [2], 4: [3, 4]},
            use_control_changes=True,
            control_change_numbers=[96],
            use_pitch_bends=True,
            use_sustain_pedals=True,
        )
    )
    score = Score(16)
    score.time_signatures = [
        TimeSignature(0, 3, 16),
        TimeSignature(12, 2, 2),
        TimeSignature(76, 3, 4),
    ]
    times = [0, 15, 16, 17, 47, 49, 77, 78, 83]
    score.tracks = [
        Track(
            notes=[Note(time, 8, 60 + index, 80) for index, time in enumerate(times)],
            controls=[ControlChange(time, 96, 0) for time in times],
            pitch_bends=[PitchBend(time, 8191) for time in times],
            pedals=[Pedal(15, 1), Pedal(49, 30)],
        )
    ]
    preprocessed = tokenizer.preprocess_score(score)
    track = preprocessed.tracks[0]
    expected_times = [0, 12, 12, 20, 44, 52, 76, 76, 84]
    assert [note.time for note in track.notes] == expected_times
    # Quantization must preserve repeated CC commands but deduplicate pitch bends.
    assert [control.time for control in track.controls] == expected_times
    assert [bend.time for bend in track.pitch_bends] == [0, 12, 20, 44, 52, 76, 84]
    assert list(track.pedals) == [Pedal(12, 8), Pedal(52, 28)]
    decoded = tokenizer.decode(tokenizer.encode(preprocessed, no_preprocess_score=True))
    # Decoding pedals also emits CC64; the explicit CC96 stream must stay intact.
    assert [cc for cc in decoded.tracks[0].controls if cc.number == 96] == list(
        track.controls
    )
    assert decoded.tracks[0].pitch_bends == track.pitch_bends
    assert decoded.tracks[0].pedals == track.pedals


@pytest.mark.parametrize("tokenization", ["BEAT", "MIDILike"])
def test_note_endpoints_across_time_signatures(tokenization: str) -> None:
    """Keep unsorted offsets attached to notes and expand collapsed notes."""
    tokenizer = getattr(miditok, tokenization)(
        miditok.TokenizerConfig(
            beat_res={(0, 32): 4},
            num_velocities=127,
            use_time_signatures=True,
            time_signature_range={16: [3], 2: [2], 4: [3, 4]},
        )
    )
    score = Score(16)
    score.time_signatures = [
        TimeSignature(0, 3, 16),
        TimeSignature(12, 2, 2),
        TimeSignature(76, 3, 4),
    ]
    score.tracks = [
        Track(
            notes=[
                Note(1, 86, 60, 80),
                Note(10, 5, 62, 80),
                Note(15, 1, 64, 80),
                Note(75, 1, 65, 80),
            ]
        )
    ]
    expected_notes = [
        Note(1, 87, 60, 80),
        Note(10, 2, 62, 80),
        Note(12, 8, 64, 80),
        Note(76, 4, 65, 80),
    ]
    preprocessed = tokenizer.preprocess_score(score)
    assert list(preprocessed.tracks[0].notes) == expected_notes
    assert tokenizer.preprocess_score(preprocessed) == preprocessed
    decoded = tokenizer.decode(tokenizer.encode(preprocessed, no_preprocess_score=True))
    assert sorted(decoded.tracks[0].notes, key=lambda note: note.time) == expected_notes


@pytest.mark.parametrize("tokenization", ["BEAT", "REMI", "TSD", "MIDILike", "MuMIDI"])
def test_preprocess_same_program_tracks(tokenization: str) -> None:
    """Keep BEAT tracks separate while other single-stream tokenizers still merge."""
    tokenizer = getattr(miditok, tokenization)(
        miditok.TokenizerConfig(use_programs=True)
    )
    score = Score(480)
    score.tracks = [
        Track(notes=[Note(0, 480, 60, 80)]),
        Track(notes=[Note(0, 480, 64, 80)]),
    ]
    preprocessed = tokenizer.preprocess_score(score)
    assert len(preprocessed.tracks) == (2 if tokenization == "BEAT" else 1)
    assert sum(len(track.notes) for track in preprocessed.tracks) == 2
    assert len(score.tracks) == 2


@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize("file_path", MIDI_PATHS_ALL, ids=lambda p: p.name)
def test_preprocess(tokenization: str, file_path: Path) -> None:
    r"""
    Check that a second preprocessing doesn't alter the MIDI anymore.

    :param tokenization: name of the tokenizer class.
    :param file_path: paths to MIDI file to test.
    """
    # Creates tokenizer
    tok_config = miditok.TokenizerConfig(**CONFIG_KWARGS)
    tokenizer = getattr(miditok, tokenization)(tok_config)

    # Preprocess original file, and once again on the already preprocessed file
    score = Score(file_path)
    score_processed1 = tokenizer.preprocess_score(score)
    score_processed2 = tokenizer.preprocess_score(score_processed1)

    # The second preprocess shouldn't do anything
    assert score_processed1 == score_processed2


@pytest.mark.parametrize("tokenization", ["REMI", "TSD", "MIDILike", "PerTok"])
@pytest.mark.parametrize("use_programs", [False, True])
def test_control_changes_deduplication(tokenization: str, use_programs: bool) -> None:
    """Deduplicate states while preserving barriers, transitions and original times."""
    params = {
        "use_control_changes": True,
        "control_change_numbers": [7, 64, 66, 67],
        "control_change_n_bins": 3,
        "use_programs": use_programs,
    }
    adjust_tok_params_for_tests(tokenization, params)
    tokenizer = getattr(miditok, tokenization)(miditok.TokenizerConfig(**params))
    score = Score(4800)
    controls = [
        # Equal quantized levels collapse; different values stay in order.
        (0, 7, 62),
        (0, 7, 65),
        (0, 7, 65),
        (0, 7, 127),
        (0, 7, 64),
        # Filtered LSB, reset and unknown controllers still break adjacency.
        (0, 39, 17),
        (0, 7, 64),
        (0, 121, 0),
        (0, 7, 64),
        (0, 3, 0),
        (0, 7, 64),
        # Repeated pedal states collapse, but on/off/on transitions survive.
        (0, 64, 127),
        (0, 64, 127),
        (0, 64, 0),
        (0, 64, 127),
        (0, 66, 63),
        (0, 66, 48),
        (0, 66, 64),
        (0, 66, 127),
        (0, 67, 48),
        (0, 67, 62),
        # These distinct ticks become equal during initial score resampling.
        (1, 7, 64),
        (2, 7, 64),
    ]
    score.tracks.append(
        Track(
            notes=[Note(0, 4800, 60, 100)],
            controls=[ControlChange(*control) for control in controls],
        )
    )
    # Track merging must not introduce additional deduplication.
    score.tracks.append(
        Track(notes=[Note(0, 4800, 67, 100)], controls=[ControlChange(2, 7, 64)])
    )
    original_score = score.copy()
    preprocessed = tokenizer.preprocess_score(score)
    assert score == original_score
    expected_values = [
        (7, 64),
        (7, 127),
        (7, 64),
        (7, 64),
        (7, 64),
        (7, 64),
        (64, 127),
        (64, 0),
        (64, 127),
        (66, 0),
        (66, 127),
        (67, 64),
        (7, 64),
        (7, 64),
    ]
    if use_programs:
        expected_values.append((7, 64))
    expected_controls = [
        ControlChange(0, number, value) for number, value in expected_values
    ]
    assert list(preprocessed.tracks[0].controls) == expected_controls
    decoded = tokenizer.decode(tokenizer.encode(preprocessed, no_preprocess_score=True))
    assert list(decoded.tracks[0].controls) == expected_controls


@pytest.mark.parametrize(
    "control_number",
    [0, 3, 6, 32, 38, 39, 84, 88, 96, 97, 98, 99, 100, 101, *range(120, 128)],
)
def test_control_changes_preserve_commands(control_number: int) -> None:
    """Preserve repeated commands, selectors, fine-resolution and unknown controls."""
    tokenizer = miditok.REMI(
        miditok.TokenizerConfig(
            use_control_changes=True, control_change_numbers=[control_number]
        )
    )
    score = Score(480)
    score.tracks.append(
        Track(controls=[ControlChange(0, control_number, 0) for _ in range(2)])
    )
    preprocessed = tokenizer.preprocess_score(score)
    assert list(preprocessed.tracks[0].controls) == list(score.tracks[0].controls)
