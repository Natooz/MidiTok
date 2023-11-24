#!/usr/bin/python3 python

"""Tests to create tokenizers, save their config, and load it back.
If all went well the tokenizer should be identical.

"""

import pytest

import miditok

from .utils import ALL_TOKENIZATIONS, TEST_DIR

ADDITIONAL_TOKENS_TEST = {
    "use_chords": False,  # set False to speed up tests as it takes some time on maestro MIDIs
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "beat_res_rest": {(0, 16): 4},
    "nb_tempos": 32,
    "tempo_range": (40, 250),
}


@pytest.mark.parametrize("tokenization", ALL_TOKENIZATIONS)
def test_saving_loading_tokenizer_config(tokenization: str):
    config1 = miditok.TokenizerConfig()
    config1.save_to_json(TEST_DIR / "configs" / f"tok_conf_{tokenization}.json")

    config2 = miditok.TokenizerConfig.load_from_json(
        TEST_DIR / "configs" / f"tok_conf_{tokenization}.json"
    )

    assert config1 == config2
    config1.pitch_range = (0, 777)
    assert config1 != config2


@pytest.mark.parametrize("tokenization", ALL_TOKENIZATIONS)
def test_saving_loading_tokenizer(tokenization: str):
    r"""Tests to create tokenizers, save their config, and load it back.
    If all went well the tokenizer should be identical.
    """
    tokenizer_config = miditok.TokenizerConfig(**ADDITIONAL_TOKENS_TEST)
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=tokenizer_config
    )
    tokenizer.save_params(TEST_DIR / "configs" / f"{tokenization}.txt")

    tokenizer2: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        params=TEST_DIR / "configs" / f"{tokenization}.txt"
    )
    assert tokenizer == tokenizer2
    if tokenization == "Octuple":
        tokenizer.vocab[0]["PAD_None"] = 8
        assert tokenizer != tokenizer2
