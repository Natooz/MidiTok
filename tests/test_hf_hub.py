#!/usr/bin/python3 python

"""Test the Hugging Face Hub integration: pushing and retrieving MidiTok tokenizers to and from the hub.

"""

from miditok import REMI, TSD


def test_push_and_load_to_hf_hub(hf_token: str):
    tokenizer = REMI()
    tokenizer.push_to_hub("Natooz/MidiTok-tests", private=True, token=hf_token)

    tokenizer2 = REMI.from_pretrained("Natooz/MidiTok-tests", token=hf_token)
    assert tokenizer == tokenizer2


def test_from_pretrained_local():
    # Here using paths to directories
    tokenizer = TSD()
    tokenizer.save_pretrained("tests/tokenizer_confs")
    tokenizer2 = TSD.from_pretrained("tests/tokenizer_confs")
    assert tokenizer == tokenizer2


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Get Hugging Face token for tests")
    parser.add_argument("--hf-token", type=str, help="", required=False, default=None)
    args = vars(parser.parse_args())

    test_from_pretrained_local()
    test_push_and_load_to_hf_hub(args["hf_token"])
