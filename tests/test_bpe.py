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
    tokenizations = ["REMI"]  # TODO remove line
    data_path = Path(data_path)
    files = list(data_path.glob("**/*.mid"))

    # Creates tokenizers and computes BPE (build voc)
    first_tokenizers = []
    for tokenization in tokenizations:
        add_tokens = deepcopy(ADDITIONAL_TOKENS_TEST)
        tokenizer = getattr(miditok, tokenization)(
            beat_res=BEAT_RES_TEST, additional_tokens=add_tokens
        )
        # tokenizer.tokenize_midi_dataset(files, Path("tests", "test_results", tokenization))  # TODO remettre
        tokenizer.learn_bpe(
            vocab_size=400,
            tokens_paths=Path("tests", "test_results", tokenization).glob("**/*.json"),
            files_lim=None,
            load_all_token_files_once=True,
            start_from_empty_voc=True,
        )
        # tokens = tokenizer.midi_to_tokens(deepcopy(MidiFile(files[0])))[0]
        tokenizer.save_params(Path("tests", "test_results", f"{tokenization}_bpe", "config.txt"))
        first_tokenizers.append(tokenizer)

        test_id_to_token = {id_: tokenizer._vocab_base_byte_to_token[byte_]
                            for id_, byte_ in tokenizer._vocab_base_id_to_byte.items()}
        vocab_inv = {v: k for k, v in tokenizer._vocab_base.items()}
        assert test_id_to_token == vocab_inv, \
            "Vocabulary inversion failed, something might be wrong with the way they are built"

        for file_path in files:
            tokens = tokenizer.midi_to_tokens(deepcopy(MidiFile(file_path)))[0]
            to_tok = tokenizer._bytes_to_tokens(tokens.bytes)
            to_id = tokenizer._tokens_to_ids(to_tok)
            to_by = tokenizer._ids_to_bytes(to_id, as_one_str=True)
            assert all([to_by == tokens.bytes, to_tok == tokens.tokens, to_id == tokens.ids]), \
                "Conversion between tokens / bytes / ids failed, something might be wrong in vocabularies"

    # Reload (test) tokenizer from the saved config file
    tokenizers = []
    for i, tokenization in enumerate(tokenizations):
        tokenizers.append(
            getattr(miditok, tokenization)(
                params=Path("tests", "test_results", f"{tokenization}_bpe", "config.txt")
            )
        )
        assert tokenizers[i] == first_tokenizers[i], \
            f"Saving and reloading tokenizer failed. The reloaded tokenizer is different from the first one."

    at_least_one_error = False

    tok_times = []
    for i, file_path in enumerate(tqdm(files, desc="Testing BPE")):
        # Reads the midi
        midi = MidiFile(file_path)

        for tokenization, tokenizer in zip(tokenizations, tokenizers):
            # Convert the track in tokens
            t0 = time()
            tokens = tokenizer.midi_to_tokens(deepcopy(midi))[0]  # with BPE
            t1 = time() - t0
            tok_times.append(t1)
            with open(
                Path("tests", "test_results", tokenization, f"{file_path.stem}.json")
            ) as json_file:
                tokens_no_bpe = json.load(json_file)["ids"][0]  # no BPE
            tokens_no_bpe2 = tokenizer.decompose_bpe(deepcopy(tokens))  # BPE decomposed
            with open(
                Path("tests", "test_results", f"{tokenization}_bpe", f"{file_path.stem}.json")
            ) as json_file:
                saved_tokens = json.load(json_file)["ids"][
                    0
                ]  # with BPE, saved after creating vocab
            saved_tokens_decomposed = tokenizer.decompose_bpe(deepcopy(saved_tokens))
            no_error_bpe = tokens == saved_tokens
            no_error = tokens_no_bpe == tokens_no_bpe2 == saved_tokens_decomposed
            if not no_error or not no_error_bpe:
                at_least_one_error = True
                print(f"error for {tokenization} and {file_path.name}")
    print(f"Mean tokenization time: {sum(tok_times) / len(tok_times):.2f}")
    assert not at_least_one_error


if __name__ == "__main__":
    test_bpe_conversion()
