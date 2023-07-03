#!/usr/bin/python3 python

"""Tests to create tokenizers, save their config, and load it back.
If all went well the tokenizer should be identical.

"""

import miditok


ADDITIONAL_TOKENS_TEST = {
    "use_chords": False,  # set False to speed up tests as it takes some time on maestro MIDIs
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "rest_range": (4, 16),
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": (16, 2),
}
tokenizations = [
    "MIDILike",
    "TSD",
    "Structured",
    "REMI",
    "REMIPlus",
    "CPWord",
    "Octuple",
    "OctupleMono",
    "MuMIDI",
]


def test_saving_loading_tokenizer_config():
    for tokenization in tokenizations:
        config1 = miditok.TokenizerConfig()
        config1.save_to_json(f"./tests/configs/tok_conf_{tokenization}.json")

        config2 = miditok.TokenizerConfig.load_from_json(
            f"./tests/configs/tok_conf_{tokenization}.json"
        )

        assert config1 == config2
        config1.pitch_range = (0, 777)
        assert config1 != config2


def test_saving_loading_tokenizer():
    r"""Tests to create tokenizers, save their config, and load it back.
    If all went well the tokenizer should be identical.
    """

    for tokenization in tokenizations:
        tokenizer_config = miditok.TokenizerConfig(**ADDITIONAL_TOKENS_TEST)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            tokenizer_config=tokenizer_config
        )
        tokenizer.save_params(f"./tests/configs/{tokenization}.txt")

        tokenizer2: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            params=f"./tests/configs/{tokenization}.txt"
        )
        assert tokenizer == tokenizer2
        if tokenization == "Octuple":
            tokenizer.vocab[0]["PAD_None"] = 8
            assert tokenizer != tokenizer2


if __name__ == "__main__":
    test_saving_loading_tokenizer_config()
    test_saving_loading_tokenizer()
