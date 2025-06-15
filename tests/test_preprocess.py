"""Tests on the preprocessing steps of music files, before tokenization."""

from pathlib import Path

import pytest
from symusic import Score

import miditok

from .utils_tests import MIDI_PATHS_ALL

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
def test_preprocess(tokenization: str, file_path: Path):
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
