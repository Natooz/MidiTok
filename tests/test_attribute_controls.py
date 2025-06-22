"""Test methods."""

from __future__ import annotations

from random import seed
from typing import TYPE_CHECKING, Literal

import pytest
from symusic import Score

import miditok
from miditok.attribute_controls import create_random_ac_indexes

from .utils_tests import (
    BARS_RANDOM_RATIO_RANGE,
    MIDI_PATHS_ALL,
    MIDI_PATHS_ONE_TRACK,
    SEED,
    TRACKS_RANDOM_RATIO_RANGE,
    check_control_tokens_are_well_inserted,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

TOKENIZATIONS = ["REMI", "TSD", "MMM"]
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    "base_tokenizer": "REMI",
    "ac_polyphony_track": True,
    "ac_polyphony_bar": True,
    "ac_pitch_class_bar": True,
    "ac_note_density_track": True,
    "ac_note_density_bar": True,
    "ac_note_duration_bar": True,
    "ac_note_duration_track": True,
    "ac_repetition_track": True,
}
VOCAB_SIZE = 2000
NUM_ADDITIONAL_TOKENS_SECOND_TRAINING = 400
WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BAR = 500  # higher than default MidiTok values
WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BEAT = 150


@pytest.mark.parametrize("file_path", MIDI_PATHS_ALL, ids=lambda path: path.name)
@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize(
    "random_tracks_idx",
    [False, True],
    ids=lambda r: "rand_tracks" if r else "all_tracks",
)
@pytest.mark.parametrize(
    "random_bars_idx", [False, True], ids=lambda r: "rand_bars" if r else "all_bars"
)
def test_attribute_controls_computation(
    file_path: Path,
    tokenization: str,
    random_tracks_idx: bool,
    random_bars_idx: bool,
) -> None:
    tokenizer_params = TOKENIZER_PARAMS

    tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**tokenizer_params)
    )
    score = Score(file_path)
    score = tokenizer.preprocess_score(score)

    # Set attribute controls indexes
    seed(SEED)
    tracks_idx_ratio = (0, 1) if random_tracks_idx else 1
    bars_idx_ratio = (0, 1) if random_bars_idx else 1
    ac_indexes = create_random_ac_indexes(
        score,
        tokenizer.attribute_controls,
        tracks_idx_ratio,
        bars_idx_ratio,
    )

    # Tokenize Score with attribute controls injected
    tokens = tokenizer.encode(
        score, no_preprocess_score=True, attribute_controls_indexes=ac_indexes
    )

    # Check for errors
    injection_errors = check_control_tokens_are_well_inserted(
        tokenizer, score, tokens, ac_indexes
    )
    assert len(injection_errors) == 0


@pytest.mark.parametrize("tokenization", TOKENIZATIONS)
@pytest.mark.parametrize("model", ["BPE"])
@pytest.mark.parametrize(
    "encode_ids_split",
    ["no", "bar", "beat"],
    ids=lambda s: f"{s}_split",
)
@pytest.mark.parametrize("files_paths", [MIDI_PATHS_ONE_TRACK], ids=lambda _: "")
@pytest.mark.parametrize("vocab_size", [VOCAB_SIZE], ids=lambda s: f"vocab size {s}")
def test_tokenizer_training_and_encoding_decoding(
    tokenization: str,
    model: Literal["BPE", "Unigram", "WordPiece"],
    encode_ids_split: Literal["bar", "beat", "no"],
    files_paths: Sequence[Path],
    vocab_size: int,
):
    r"""
    Train a tokenizer to make sure the training iterator works with attribute controls.

    :param files_paths: list of paths of music files to use for the tests.
    :param encode_ids_split: type of token ids split before encoding/training.
    """
    if encode_ids_split == "no" and model == "WordPiece":
        pytest.skip(f"Skipping training with {model} and {encode_ids_split} split")

    # Creates tokenizers
    TOKENIZER_PARAMS["encode_ids_split"] = encode_ids_split
    tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**TOKENIZER_PARAMS)
    )

    training_kwargs = {}
    if model == "WordPiece":
        training_kwargs["max_input_chars_per_word"] = (
            WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BAR
            if encode_ids_split == "bar"
            else WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BEAT
        )

    # Train the tokenizer
    training_iterator = miditok.TokTrainingIterator(
        tokenizer, files_paths, TRACKS_RANDOM_RATIO_RANGE, BARS_RANDOM_RATIO_RANGE
    )
    tokenizer.train(
        vocab_size=vocab_size + NUM_ADDITIONAL_TOKENS_SECOND_TRAINING,
        model=model,
        iterator=training_iterator,
        **training_kwargs,
    )
