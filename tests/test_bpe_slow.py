#!/usr/bin/python3 python

""" One track test file
This test method is to be used with MIDI files of one track (like the maestro dataset).
It is mostly useful to measure the performance of encodings where time is based on
time shifts tokens, as these files usually don't contain tracks with very long pauses,
i.e. long duration / time-shift values probably out of range of the tokenizer's vocabulary.

NOTE: encoded tracks has to be compared with the quantized original track.

"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union
import json
from time import time

import miditok
from miditoolkit import MidiFile
from tqdm import tqdm

# Special beat res for test, up to 64 beats so the duration and time-shift values are
# long enough for MIDI-Like and Structured encodings, and with a single beat resolution
BEAT_RES_TEST = {(0, 4): 8, (4, 12): 4}
ADDITIONAL_TOKENS_TEST = {
    "Chord": False,  # set false to speed up tests as it takes some time on maestro MIDIs
    "Rest": True,
    "Tempo": True,
    "TimeSignature": True,
    "Program": False,
    "rest_range": (4, 16),
    "nb_tempos": 32,
    "tempo_range": (40, 250),
    "time_signature_range": (16, 2),
}


def test_bpe_conversion(
    data_path: Union[str, Path, PurePath] = "./tests/Maestro_MIDIs"
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back to MIDI files.
    The converted back MIDI files should identical to original one, expect with note starting and ending
    times quantized, and maybe a some duplicated notes removed

    :param data_path: root path to the data to test
    """
    tokenizations = ["Structured", "REMI", "MIDILike", "TSD"]
    data_path = Path(data_path)
    files = list(data_path.glob("**/*.mid"))

    # Creates tokenizers and computes BPE (build voc)
    for tokenization in tokenizations:
        add_tokens = deepcopy(ADDITIONAL_TOKENS_TEST)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            beat_res=BEAT_RES_TEST, additional_tokens=add_tokens
        )
        tokenizer.tokenize_midi_dataset(
            files, Path("tests", "test_results", tokenization)
        )
        tokenizer.learn_bpe_slow(
            Path("tests", "test_results", tokenization),
            len(tokenizer.vocab) + 30,
            out_dir=Path("tests", "test_results", f"{tokenization}_bpe"),
            files_lim=None,
            save_converted_samples=True,
        )

    # Reload (test) tokenizer from the saved config file
    tokenizers = []
    for i, tokenization in enumerate(tokenizations):
        tokenizers.append(
            getattr(miditok, tokenization)(
                params=Path("tests", "test_results", f"{tokenization}_bpe")
                / "config.txt"
            )
        )

    at_least_one_error = False

    tok_times = []
    for i, file_path in enumerate(tqdm(files, desc="Testing BPE")):
        # Reads the midi
        midi = MidiFile(file_path)

        for tokenization, tokenizer in zip(tokenizations, tokenizers):
            # Convert the track in tokens
            t0 = time()
            tokens = tokenizer.midi_to_tokens(deepcopy(midi))  # with BPE
            t1 = time() - t0
            if not tokenizer.unique_track:
                tokens = tokens[0]
            tok_times.append(t1)
            with open(
                Path("tests", "test_results", tokenization, f"{file_path.stem}.json")
            ) as json_file:
                tokens_no_bpe = json.load(json_file)["ids"]  # no BPE
            if not tokenizer.unique_track:
                tokens_no_bpe = tokens_no_bpe[0]
            tokens_no_bpe = miditok.TokSequence(ids=tokens_no_bpe, ids_bpe_encoded=True)
            tokens_bpe_decomposed = deepcopy(tokens)  # BPE decomposed
            tokenizer.decode_bpe(tokens_bpe_decomposed)  # BPE decomposed
            with open(
                Path(
                    "tests",
                    "test_results",
                    f"{tokenization}_bpe",
                    f"{file_path.stem}.json",
                )
            ) as json_file:
                saved_tokens = json.load(json_file)[
                    "ids"
                ]  # with BPE, saved after creating vocab
            if not tokenizer.unique_track:
                saved_tokens = saved_tokens[0]
            saved_tokens = miditok.TokSequence(ids=saved_tokens, ids_bpe_encoded=True)
            saved_tokens_decomposed = deepcopy(saved_tokens)
            tokenizer.decode_bpe(saved_tokens_decomposed)
            no_error_bpe = tokens == saved_tokens
            no_error = tokens_no_bpe == tokens_bpe_decomposed == saved_tokens_decomposed
            if not no_error or not no_error_bpe:
                at_least_one_error = True
                print(f"error for {tokenization} and {file_path.name}")
    print(f"Mean tokenization time: {sum(tok_times) / len(tok_times):.2f}")
    assert not at_least_one_error


if __name__ == "__main__":
    test_bpe_conversion()
