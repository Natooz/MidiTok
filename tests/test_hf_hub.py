#!/usr/bin/python3 python

"""Test the Hugging Face Hub integration: pushing and retrieving MidiTok tokenizers to
and from the hub.

"""

from pathlib import Path
from time import sleep

from huggingface_hub.utils._errors import HfHubHTTPError

from miditok import REMI, TSD

MAX_NUM_TRIES_HF_PUSH = 5
NUM_SECONDS_RETRY = 8


def test_push_and_load_to_hf_hub(hf_token: str):
    tokenizer = REMI()
    num_tries = 0
    while num_tries < MAX_NUM_TRIES_HF_PUSH:
        try:
            tokenizer.push_to_hub("Natooz/MidiTok-tests", private=True, token=hf_token)
        except HfHubHTTPError as e:  # noqa: PERF203
            if e.response.status_code in [500, 412, 429]:
                num_tries += 1
                sleep(NUM_SECONDS_RETRY)
            else:
                num_tries = MAX_NUM_TRIES_HF_PUSH

    tokenizer2 = REMI.from_pretrained("Natooz/MidiTok-tests", token=hf_token)
    assert tokenizer == tokenizer2


def test_from_pretrained_local(tmp_path: Path):
    # Here using paths to directories
    tokenizer = TSD()
    tokenizer.save_pretrained(tmp_path)
    tokenizer2 = TSD.from_pretrained(tmp_path)
    assert tokenizer == tokenizer2
