"""Test the integration of the Hugging Face Hub."""

from __future__ import annotations

import warnings
from time import sleep
from typing import TYPE_CHECKING

import pytest
from huggingface_hub.utils import HfHubHTTPError

import miditok

if TYPE_CHECKING:
    from pathlib import Path

MAX_NUM_TRIES_HF_PUSH = 3
NUM_SECONDS_RETRY = 8

AUTO_TOKENIZER_CASES = [
    # ("class_name", "save_path", "class_name_assert")
    ("REMI", "rem", "REMI"),
    ("REMI", "rem2", "TSD"),
    ("TSD", "tsd", "TSD"),
]


def test_push_and_load_to_hf_hub(hf_token: str):
    tokenizer = miditok.REMI(
        miditok.TokenizerConfig(num_velocities=62, pitch_range=(12, 44))
    )
    num_tries = 0
    while num_tries < MAX_NUM_TRIES_HF_PUSH:
        try:
            tokenizer.push_to_hub("Natooz/MidiTok-tests", private=True, token=hf_token)
        except HfHubHTTPError as e:
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

    tokenizer2 = miditok.REMI.from_pretrained("Natooz/MidiTok-tests", token=hf_token)
    assert tokenizer == tokenizer2


def test_from_pretrained_local(tmp_path: Path):
    # Here using paths to directories
    tokenizer = miditok.TSD()
    tokenizer.save_pretrained(tmp_path)
    tokenizer2 = miditok.TSD.from_pretrained(tmp_path)
    assert tokenizer == tokenizer2


@pytest.mark.parametrize("params_case", AUTO_TOKENIZER_CASES)
def test_autotokenizer(tmp_path: Path, params_case: tuple[str, str, str]):
    tok_class, save_path, tok_class2 = params_case

    tokenizer = getattr(miditok, tok_class)()
    tokenizer.save_pretrained(tmp_path / save_path)
    tokenizer2 = getattr(miditok, tok_class2)(
        params=tmp_path / save_path / "tokenizer.json"
    )

    assert (tokenizer == tokenizer2) == (tok_class == tok_class2)
