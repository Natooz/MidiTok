#!/usr/bin/python3 python

"""Test the Hugging Face Hub integration: pushing and retrieving MidiTok tokenizers to and from the hub.

"""

from pathlib import Path

from miditok import REMI, TSD


def test_push_and_load_to_hf_hub(hf_token: str):
    # TODO handle errors, retry if 500/412?
    tokenizer = REMI()
    tokenizer.push_to_hub("Natooz/MidiTok-tests", private=True, token=hf_token)

    tokenizer2 = REMI.from_pretrained("Natooz/MidiTok-tests", token=hf_token)
    assert tokenizer == tokenizer2


def test_from_pretrained_local(tmp_path: Path):
    # Here using paths to directories
    tokenizer = TSD()
    tokenizer.save_pretrained(tmp_path)
    tokenizer2 = TSD.from_pretrained(tmp_path)
    assert tokenizer == tokenizer2
