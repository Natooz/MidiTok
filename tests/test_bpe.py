#!/usr/bin/python3 python

"""Tests Fast BPE encoding - decoding, as well as saving and loading tokenizers with
BPE.
"""

from __future__ import annotations

import random
from copy import copy, deepcopy
from pathlib import Path
from time import time
from typing import Sequence

import pytest
from symusic import Score
from tqdm import tqdm

import miditok

from .utils import (
    MIDI_PATHS_ONE_TRACK,
    SEED,
    TOKENIZATIONS_BPE,
    TOKENIZER_CONFIG_KWARGS,
    adjust_tok_params_for_tests,
)

default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
default_params.update(
    {
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
    }
)


@pytest.mark.parametrize("tokenization", TOKENIZATIONS_BPE)
def test_bpe_conversion(
    tokenization: str,
    tmp_path: Path,
    midi_paths: Sequence[str | Path] | None = None,
    seed: int = SEED,
):
    r"""Reads a few MIDI files, convert them into token sequences, convert them back
    to MIDI files. The converted back MIDI files should identical to original one,
    expect with note starting and ending times quantized, and maybe a some duplicated
    notes removed

    :param tokenization: name of the tokenizer class to test.
    :param midi_paths: list of paths of MIDI files to use for the tests.
    :param seed: seed.
    """
    if midi_paths is None:
        midi_paths = MIDI_PATHS_ONE_TRACK
    random.seed(seed)

    # Creates tokenizers and computes BPE (build voc)
    first_samples_bpe = []
    params_ = deepcopy(default_params)
    adjust_tok_params_for_tests(tokenization, params_)
    tokenizer: miditok.MIDITokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params_)
    )
    tokenizer.learn_bpe(
        vocab_size=len(tokenizer) + 400,
        files_paths=midi_paths,
        start_from_empty_voc=True,
    )
    tokenizer.save_params(tmp_path / "bpe_config.txt")

    test_id_to_token = {
        id_: tokenizer._vocab_base_byte_to_token[byte_]
        for id_, byte_ in tokenizer._vocab_base_id_to_byte.items()
    }
    vocab_inv = {v: k for k, v in tokenizer._vocab_base.items()}
    assert (
        test_id_to_token == vocab_inv
    ), "Vocabulary inversion failed, something is wrong with the way they are built"

    for file_path in tqdm(
        midi_paths, desc=f"Checking BPE tok / detok ({tokenization})"
    ):
        tokens = tokenizer(file_path, apply_bpe_if_possible=False)
        if not tokenizer.one_token_stream:
            tokens = tokens[0]
        to_tok = tokenizer._bytes_to_tokens(tokens.bytes)
        to_id = tokenizer._tokens_to_ids(to_tok)
        to_by = tokenizer._ids_to_bytes(to_id, as_one_str=True)
        assert all(
            [to_by == tokens.bytes, to_tok == tokens.tokens, to_id == tokens.ids]
        ), (
            "Conversion between tokens / bytes / ids failed, something is wrong in"
            "vocabularies"
        )

        tokenizer.apply_bpe(tokens)
        first_samples_bpe.append(tokens)

    # Reload (test) tokenizer from the saved config file
    tokenizer_reloaded = getattr(miditok, tokenization)(
        params=tmp_path / "bpe_config.txt"
    )
    assert tokenizer_reloaded == tokenizer, (
        "Saving and reloading tokenizer failed. The reloaded tokenizer is different"
        "from the first one."
    )

    # Unbatched BPE
    at_least_one_error = False
    tok_time = 0
    for i, file_path in enumerate(tqdm(midi_paths, desc="Testing BPE unbatched")):
        midi = Score(file_path)
        tokens_no_bpe = tokenizer(copy(midi), apply_bpe_if_possible=False)
        if not tokenizer.one_token_stream:
            tokens_no_bpe = tokens_no_bpe[0]
        tokens_bpe = deepcopy(tokens_no_bpe)  # with BPE

        t0 = time()
        tokenizer.apply_bpe(tokens_bpe)
        tok_time += time() - t0

        tokens_bpe_decoded = deepcopy(tokens_bpe)
        tokenizer.decode_bpe(tokens_bpe_decoded)  # BPE decomposed
        if tokens_bpe != first_samples_bpe[i]:
            at_least_one_error = True
            print(
                f"Error with BPE for {tokenization} and {file_path.name}: "
                f"BPE encoding failed after tokenizer reload"
            )
        if tokens_no_bpe != tokens_bpe_decoded:
            at_least_one_error = True
            print(
                f"Error with BPE for {tokenization} and {file_path.name}: encoding -"
                f"decoding test failed"
            )
    print(
        f"BPE encoding time un-batched: {tok_time:.2f} (mean:"
        f"{tok_time / len(midi_paths):.4f})"
    )
    assert not at_least_one_error

    # Batched BPE
    at_least_one_error = False
    samples_no_bpe = []
    for file_path in tqdm(midi_paths, desc="Testing BPE batched"):
        # Reads the midi
        midi = Score(file_path)
        tokens_no_bpe = tokenizer(midi, apply_bpe_if_possible=False)
        if not tokenizer.one_token_stream:
            samples_no_bpe.append(tokens_no_bpe[0])
        else:
            samples_no_bpe.append(tokens_no_bpe)
    samples_bpe = deepcopy(samples_no_bpe)
    t0 = time()
    tokenizer.apply_bpe(samples_bpe)
    tok_time = time() - t0
    samples_bpe_decoded = deepcopy(samples_bpe)
    tokenizer.decode_bpe(samples_bpe_decoded)  # BPE decomposed
    for sample_bpe, sample_bpe_first in zip(samples_bpe, first_samples_bpe):
        if sample_bpe != sample_bpe_first:
            at_least_one_error = True
            print(
                f"Error with BPE for {tokenization}: BPE encoding failed after"
                f"tokenizer reload"
            )
    for sample_no_bpe, sample_bpe_decoded in zip(samples_no_bpe, samples_bpe_decoded):
        if sample_no_bpe != sample_bpe_decoded:
            at_least_one_error = True
            print(f"Error with BPE for {tokenization}: encoding - decoding test failed")
    print(
        f"BPE encoding time batched: {tok_time:.2f} (mean:"
        f"{tok_time / len(midi_paths):.4f})"
    )
    assert not at_least_one_error
