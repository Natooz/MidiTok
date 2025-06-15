"""Tests for the saving/loading methods of tokenizers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest

import miditok

from .utils_tests import (
    ALL_TOKENIZATIONS,
    MAX_BAR_EMBEDDING,
    MIDI_PATHS_MULTITRACK,
    MIDI_PATHS_ONE_TRACK,
)

if TYPE_CHECKING:
    from pathlib import Path

ADDITIONAL_TOKENS_TEST = {
    "use_chords": False,  # False to speed up tests
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "beat_res_rest": {(0, 16): 4},
    "num_tempos": 32,
    "tempo_range": (40, 250),
    "base_tokenizer": "TSD",
    "use_microtiming": True,
    "ticks_per_quarter": 480,
    "max_microtiming_shift": 0.25,
    "num_microtiming_bins": 110,
}

TOK_PARAMS_MULTITRACK = []
tokenizations_non_one_stream = [
    "TSD",
    "REMI",
    "MIDILike",
    "Structured",
    "CPWord",
    "Octuple",
]
for tokenization_ in ALL_TOKENIZATIONS:
    params_ = {"use_programs": True}
    if tokenization_ == "MMM":
        params_["base_tokenizer"] = "TSD"
    elif tokenization_ in ["Octuple", "MuMIDI"]:
        params_["max_bar_embedding"] = MAX_BAR_EMBEDDING
    elif tokenization_ in ["PerTok"]:
        params_["use_microtiming"] = True
        params_["ticks_per_quarter"] = 220
        params_["max_microtiming_shift"] = 0.25
        params_["num_microtiming_bins"] = 110
    TOK_PARAMS_MULTITRACK.append((tokenization_, params_))

    if tokenization_ in tokenizations_non_one_stream:
        params_tmp = deepcopy(params_)
        params_tmp["one_token_stream_for_programs"] = False
        # Disable tempos for Octuple with one_token_stream_for_programs, as tempos are
        # carried by note tokens
        if tokenization_ == "Octuple":
            params_tmp["use_tempos"] = False
        TOK_PARAMS_MULTITRACK.append((tokenization_, params_tmp))


@pytest.mark.parametrize("tokenization", ALL_TOKENIZATIONS)
def test_saving_loading_tokenizer_config(tokenization: str, tmp_path: Path):
    config1 = miditok.TokenizerConfig()
    config1.save_to_json(tmp_path / f"tok_conf_{tokenization}.json")

    config2 = miditok.TokenizerConfig.load_from_json(
        tmp_path / f"tok_conf_{tokenization}.json"
    )

    assert config1 == config2
    config1.pitch_range = (0, 777)
    assert config1 != config2


@pytest.mark.parametrize("tokenization", ALL_TOKENIZATIONS)
def test_saving_loading_tokenizer(tokenization: str, tmp_path: Path):
    r"""
    Make sure saving and loading end with the identical tokenizer.

    Create a tokenizer, save its config, and load it back.
    If all went well the reloaded tokenizer should be identical.
    """
    tokenizer_config = miditok.TokenizerConfig(**ADDITIONAL_TOKENS_TEST)
    tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=tokenizer_config
    )
    tokenizer.save(tmp_path / f"{tokenization}.txt")

    tokenizer2: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        params=tmp_path / f"{tokenization}.txt"
    )
    assert tokenizer == tokenizer2
    if tokenization == "Octuple":
        tokenizer.vocab[0]["PAD_None"] = 8
        assert tokenizer != tokenizer2


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK[:3], ids=lambda p: p.name)
@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_MULTITRACK)
def test_multitrack_midi_to_tokens_to_midi(
    file_path: Path,
    tok_params_set: tuple[str, dict[str, Any]],
    tmp_path: Path,
):
    # Create tokenizer
    tokenization, params = tok_params_set
    tokenizer: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )

    # Tokenize the file, save tokens and load them back
    tokens = tokenizer(file_path)
    tokenizer.save_tokens(tokens, tmp_path / "tokens.json")
    tokens_loaded = tokenizer.load_tokens(tmp_path / "tokens.json")

    # Assert tokens are the same
    assert tokens == tokens_loaded


@pytest.mark.parametrize("file_path", MIDI_PATHS_ONE_TRACK[:3], ids=lambda p: p.name)
def test_pertok_microtiming_tick_values(file_path: Path):
    # Create the pertok tokenizer
    cfg = miditok.TokenizerConfig(
        use_chords=False,
        use_microtiming=True,
        ticks_per_quarter=480,
        max_microtiming_shift=0.25,
        num_microtiming_bins=110,
    )
    tok = miditok.PerTok(cfg)
    # Train the tokenizer
    tok.train(files_paths=[file_path], vocab_size=1000)
    # Dump the tokenizer to a JSON
    tok.save("tmp.json")
    # Reload the tokenizer
    newtok = miditok.PerTok(params="tmp.json")
    # Should still have the microtiming_tick_values parameter
    assert hasattr(newtok, "microtiming_tick_values")
