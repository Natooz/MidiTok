"""Tests on the preprocessing steps of MIDI files, before tokenization."""

from pathlib import Path

import pytest
from symusic import Score

import miditok

from .utils import MIDI_PATHS_ALL

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
@pytest.mark.parametrize("midi_path", MIDI_PATHS_ALL)
def test_preprocess(tokenization: str, midi_path: Path):
    r"""
    Check that a second preprocessing doesn't alter the MIDI anymore.

    :param tokenization: name of the tokenizer class.
    :param midi_path: paths to MIDI file to test.
    """
    # Creates tokenizer
    tok_config = miditok.TokenizerConfig(**CONFIG_KWARGS)
    tokenizer = getattr(miditok, tokenization)(tok_config)

    # Preprocess original MIDI, and once again on the already preprocessed MIDI
    midi = Score(midi_path)
    midi_processed1 = tokenizer.preprocess_midi(midi)
    midi_processed2 = tokenizer.preprocess_midi(midi_processed1)

    # The second preprocess shouldn't do anything
    assert midi_processed1 == midi_processed2
