#!/usr/bin/python3 python

"""Tests Fast BPE encoding - decoding, as well as saving and loading tokenizers with BPE.
"""

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Union
from time import time
import random

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
    random.seed(777)
    tokenizations = ["Structured", "REMI", "REMIPlus", "MIDILike", "TSD"]
    data_path = Path(data_path)
    files = list(data_path.glob("**/*.mid"))

    # Creates tokenizers and computes BPE (build voc)
    first_tokenizers = []
    first_samples_bpe = {tok: [] for tok in tokenizations}
    for tokenization in tokenizations:
        add_tokens = deepcopy(ADDITIONAL_TOKENS_TEST)
        tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
            beat_res=BEAT_RES_TEST, additional_tokens=add_tokens
        )
        tokenizer.tokenize_midi_dataset(
            files, Path("tests", "test_results", tokenization)
        )
        tokenizer.learn_bpe(
            vocab_size=len(tokenizer) + 400,
            tokens_paths=list(
                Path("tests", "test_results", tokenization).glob("**/*.json")
            ),
            start_from_empty_voc=True,
        )
        tokenizer.save_params(
            Path("tests", "test_results", f"{tokenization}_bpe", "config.txt")
        )
        first_tokenizers.append(tokenizer)

        test_id_to_token = {
            id_: tokenizer._vocab_base_byte_to_token[byte_]
            for id_, byte_ in tokenizer._vocab_base_id_to_byte.items()
        }
        vocab_inv = {v: k for k, v in tokenizer._vocab_base.items()}
        assert (
            test_id_to_token == vocab_inv
        ), "Vocabulary inversion failed, something might be wrong with the way they are built"

        for file_path in files:
            tokens = tokenizer(file_path, apply_bpe_if_possible=False)
            if not tokenizer.unique_track:
                tokens = tokens[0]
            to_tok = tokenizer._bytes_to_tokens(tokens.bytes)
            to_id = tokenizer._tokens_to_ids(to_tok)
            to_by = tokenizer._ids_to_bytes(to_id, as_one_str=True)
            assert all(
                [to_by == tokens.bytes, to_tok == tokens.tokens, to_id == tokens.ids]
            ), "Conversion between tokens / bytes / ids failed, something might be wrong in vocabularies"

            tokenizer.apply_bpe(tokens)
            first_samples_bpe[tokenization].append(tokens)

    # Reload (test) tokenizer from the saved config file
    tokenizers = []
    for i, tokenization in enumerate(tokenizations):
        tokenizers.append(
            getattr(miditok, tokenization)(
                params=Path(
                    "tests", "test_results", f"{tokenization}_bpe", "config.txt"
                )
            )
        )
        assert (
            tokenizers[i] == first_tokenizers[i]
        ), f"Saving and reloading tokenizer failed. The reloaded tokenizer is different from the first one."

    # Unbatched BPE
    at_least_one_error = False
    tok_times = []
    for i, file_path in enumerate(tqdm(files, desc="Testing BPE unbatched")):
        midi = MidiFile(file_path)

        for tokenization, tokenizer in zip(tokenizations, tokenizers):
            tokens_no_bpe = tokenizer(deepcopy(midi), apply_bpe_if_possible=False)
            if not tokenizer.unique_track:
                tokens_no_bpe = tokens_no_bpe[0]
            tokens_bpe = deepcopy(tokens_no_bpe)  # with BPE

            t0 = time()
            tokenizer.apply_bpe(tokens_bpe)
            tok_times.append(time() - t0)

            tokens_bpe_decoded = deepcopy(tokens_bpe)
            tokenizer.decode_bpe(tokens_bpe_decoded)  # BPE decomposed
            if tokens_bpe != first_samples_bpe[tokenization][i]:
                at_least_one_error = True
                print(
                    f"Error with BPE for {tokenization} and {file_path.name}: "
                    f"BPE encoding failed after tokenizer reload"
                )
            if tokens_no_bpe != tokens_bpe_decoded:
                at_least_one_error = True
                print(
                    f"Error with BPE for {tokenization} and {file_path.name}: encoding - decoding test failed"
                )
    print(f"Mean BPE encoding time unbatched: {sum(tok_times) / len(tok_times):.2f}")
    assert not at_least_one_error

    # Batched BPE
    at_least_one_error = False
    tok_times = []
    for tokenization, tokenizer in zip(tokenizations, tokenizers):
        samples_no_bpe = []
        for i, file_path in enumerate(tqdm(files, desc="Testing BPE batched")):
            # Reads the midi
            midi = MidiFile(file_path)
            if not tokenizer.unique_track:
                samples_no_bpe.append(tokenizer(midi, apply_bpe_if_possible=False)[0])
            else:
                samples_no_bpe.append(tokenizer(midi, apply_bpe_if_possible=False))

        t0 = time()
        samples_bpe = deepcopy(samples_no_bpe)
        tokenizer.apply_bpe(samples_bpe)
        tok_times.append((time() - t0) / len(files))

        samples_bpe_decoded = deepcopy(samples_bpe)
        tokenizer.decode_bpe(samples_bpe_decoded)  # BPE decomposed
        for sample_bpe, sample_bpe_first in zip(
            samples_bpe, first_samples_bpe[tokenization]
        ):
            if sample_bpe != sample_bpe_first:
                at_least_one_error = True
                print(
                    f"Error with BPE for {tokenization}: BPE encoding failed after tokenizer reload"
                )
        for sample_no_bpe, sample_bpe_decoded in zip(
            samples_no_bpe, samples_bpe_decoded
        ):
            if sample_no_bpe != sample_bpe_decoded:
                at_least_one_error = True
                print(
                    f"Error with BPE for {tokenization}: encoding - decoding test failed"
                )
    print(f"Mean BPE encoding time batched: {sum(tok_times) / len(tok_times):.2f}")
    assert not at_least_one_error


if __name__ == "__main__":
    test_bpe_conversion()
