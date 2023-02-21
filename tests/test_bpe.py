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
    "Chord": False,  # set to false to speed up tests as it takes some time on maestro MIDIs
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
    tokenizers = []
    data_path = Path(data_path)
    files = list(data_path.glob("**/*.mid"))

    # Creates tokenizers and computes BPE (build voc)
    for encoding in tokenizations:
        add_tokens = deepcopy(ADDITIONAL_TOKENS_TEST)
        tokenizers.append(
            getattr(miditok, encoding)(
                beat_res=BEAT_RES_TEST, additional_tokens=add_tokens
            )
        )
        tokenizers[-1].tokenize_midi_dataset(files, data_path / encoding)
        tokenizers[-1].learn_bpe(
            data_path / encoding,
            len(tokenizers[-1].vocab) + 60,
            out_dir=data_path / f"{encoding}_bpe",
            files_lim=None,
            save_converted_samples=True,
        )

    # Reload (test) tokenizer from the saved config file
    tokenizers = []
    for i, encoding in enumerate(tokenizations):
        tokenizers.append(
            getattr(miditok, encoding)(
                params=data_path / f"{encoding}_bpe" / "config.txt"
            )
        )

    at_least_one_error = False

    tok_times = []
    for i, file_path in enumerate(tqdm(files, desc="Testing BPE")):
        # Reads the midi
        midi = MidiFile(file_path)

        for encoding, tokenizer in zip(tokenizations, tokenizers):
            # Convert the track in tokens
            t0 = time()
            tokens = tokenizer.midi_to_tokens(deepcopy(midi))[0]  # with BPE
            t1 = time() - t0
            tok_times.append(t1)
            with open(
                data_path / f"{encoding}" / f"{file_path.stem}.json"
            ) as json_file:
                tokens_no_bpe = json.load(json_file)["tokens"][0]  # no BPE
            tokens_no_bpe2 = tokenizer.decompose_bpe(deepcopy(tokens))  # BPE decomposed
            with open(
                data_path / f"{encoding}_bpe" / f"{file_path.stem}.json"
            ) as json_file:
                saved_tokens = json.load(json_file)["tokens"][
                    0
                ]  # with BPE, saved after creating vocab
            saved_tokens_decomposed = tokenizer.decompose_bpe(deepcopy(saved_tokens))
            no_error_bpe = tokens == saved_tokens
            no_error = tokens_no_bpe == tokens_no_bpe2 == saved_tokens_decomposed
            if not no_error or not no_error_bpe:
                at_least_one_error = True
                print(f"error for {encoding} and {file_path.name}")
    print(f"Mean tokenization time: {sum(tok_times) / len(tok_times):.2f}")
    assert not at_least_one_error


if __name__ == "__main__":
    test_bpe_conversion()
