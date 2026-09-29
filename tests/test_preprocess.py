"""Tests on the preprocessing steps of music files, before tokenization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from symusic import ControlChange, Note, Score, Track

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
