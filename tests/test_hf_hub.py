#!/usr/bin/python3 python

"""Test the Hugging Face Hub integration: pushing and retrieving MidiTok tokenizers to
and from the hub.

"""
import warnings
from pathlib import Path
from time import sleep

import pytest
from huggingface_hub.utils._errors import HfHubHTTPError

from miditok import REMI, TSD, TokenizerConfig

MAX_NUM_TRIES_HF_PUSH = 5
NUM_SECONDS_RETRY = 8


def test_push_and_load_to_hf_hub(hf_token: str):
    tokenizer = REMI(TokenizerConfig(num_velocities=62, pitch_range=(12, 44)))
    num_tries = 0
    while num_tries < MAX_NUM_TRIES_HF_PUSH:
        try:
            tokenizer.push_to_hub("Natooz/MidiTok-tests", private=True, token=hf_token)
        except HfHubHTTPError as e:  # noqa: PERF203
            if e.response.status_code == 429:  # hourly quota exceeded
                # We performed to many tests, we skip it to not break the HF servers 🥲
                pytest.skip(
                    "Hugging Face hourly quota exceeded, skipping"
                    "`test_push_and_load_to_hf_hub` test."
                )
            elif e.response.status_code in [500, 412]:
                num_tries += 1
                sleep(NUM_SECONDS_RETRY)
            else:
                num_tries = MAX_NUM_TRIES_HF_PUSH

    # No skip, we rerun it if possible
    if num_tries == MAX_NUM_TRIES_HF_PUSH:
        warnings.warn("Tokenizer push failed", stacklevel=2)

    tokenizer2 = REMI.from_pretrained("Natooz/MidiTok-tests", token=hf_token)
    assert tokenizer == tokenizer2


def test_from_pretrained_local(tmp_path: Path):
    # Here using paths to directories
    tokenizer = TSD()
    tokenizer.save_pretrained(tmp_path)
    tokenizer2 = TSD.from_pretrained(tmp_path)
    assert tokenizer == tokenizer2
