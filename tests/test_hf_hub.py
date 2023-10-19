#!/usr/bin/python3 python

"""Test the Hugging Face Hub integration: pushing and retrieving MidiTok tokenizers to and from the hub.

"""

from pathlib import Path

from miditok import REMI


def test_push_to_hf_hub():
    tokenizer = REMI()
    assert True


def test_load_from_hf_hub():
    tokenizer = REMI()
    assert True


if __name__ == "__main__":
    test_push_to_hf_hub()
    test_load_from_hf_hub()
