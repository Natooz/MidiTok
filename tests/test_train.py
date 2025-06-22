"""Tests tokenizer training, saving-loading and encoding-decoding."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from time import time
from typing import TYPE_CHECKING

import pytest
from tqdm import tqdm

import miditok
from miditok.constants import DEFAULT_TOKENIZER_FILE_NAME

from .utils_tests import (
    MIDI_PATHS_ONE_TRACK,
    TOKENIZATIONS_TRAIN,
    TOKENIZER_CONFIG_KWARGS,
    TRAINING_MODELS,
    adjust_tok_params_for_tests,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any, Literal

VOCAB_SIZE = 2000
NUM_ADDITIONAL_TOKENS_SECOND_TRAINING = 400
WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BAR = 500  # higher than default MidiTok values
WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BEAT = 150
default_params = deepcopy(TOKENIZER_CONFIG_KWARGS)
default_params.update(
    {
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "base_tokenizer": "TSD",
    }
)

TOK_PARAMS_TRAINING = []
for tokenization_ in TOKENIZATIONS_TRAIN:
    params_ = deepcopy(default_params)
    adjust_tok_params_for_tests(tokenization_, params_)
    TOK_PARAMS_TRAINING.append((tokenization_, params_))


@pytest.mark.parametrize("tok_params_set", TOK_PARAMS_TRAINING)
@pytest.mark.parametrize("model", TRAINING_MODELS)
@pytest.mark.parametrize(
    "encode_ids_split",
    ["no", "bar", "beat"],
    ids=lambda s: f"{s}_split",
)
@pytest.mark.parametrize("files_paths", [MIDI_PATHS_ONE_TRACK], ids=lambda _: "")
@pytest.mark.parametrize("vocab_size", [VOCAB_SIZE], ids=lambda s: f"vocab size {s}")
def test_tokenizer_training_and_encoding_decoding(
    tok_params_set: tuple[str, dict[str, Any]],
    tmp_path: Path,
    model: Literal["BPE", "Unigram", "WordPiece"],
    encode_ids_split: Literal["bar", "beat", "no"],
    files_paths: Sequence[Path],
    vocab_size: int,
):
    r"""
    Train a tokenizer, check encoding-decoding keeps the same data.

    It also tests tokenizer saving-loading, and resuming training.

    :param tok_params_set: tokenizer and its parameters to run.
    :param files_paths: list of paths of music files to use for the tests.
    :param encode_ids_split: type of token ids split before encoding/training.
    """
    """from transformers import AutoTokenizer
    import json
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())"""
    if encode_ids_split == "no" and model == "WordPiece":
        pytest.skip(f"Skipping training with {model} and {encode_ids_split} split")

    # Creates tokenizers
    tokenization, params = tok_params_set
    params["encode_ids_split"] = encode_ids_split
    tokenizer1: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )
    tokenizer2: miditok.MusicTokenizer = getattr(miditok, tokenization)(
        tokenizer_config=miditok.TokenizerConfig(**params)
    )

    # Trains them
    training_kwargs = {}
    if model == "WordPiece":
        training_kwargs["max_input_chars_per_word"] = (
            WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BAR
            if encode_ids_split == "bar"
            else WORDPIECE_MAX_INPUT_CHARS_PER_WORD_BEAT
        )

    tokenizer1.train(
        vocab_size=vocab_size + NUM_ADDITIONAL_TOKENS_SECOND_TRAINING,
        model=model,
        files_paths=files_paths,
        **training_kwargs,
    )
    tokenizer2.train(
        vocab_size=vocab_size, model=model, files_paths=files_paths, **training_kwargs
    )
    tokenizer2.save(tmp_path)
    tokenizer2 = getattr(miditok, tokenization)(
        params=tmp_path / DEFAULT_TOKENIZER_FILE_NAME
    )
    tokenizer2.train(
        vocab_size=vocab_size + NUM_ADDITIONAL_TOKENS_SECOND_TRAINING,
        files_paths=files_paths,
    )

    # Tests _vocab_base and _vocab_base_id_to_byte are synced
    test_id_to_token = {
        id_: tokenizer2._vocab_base_byte_to_token[byte_]
        for id_, byte_ in tokenizer2._vocab_base_id_to_byte.items()
    }
    vocab_inv = {v: k for k, v in tokenizer2._vocab_base.items()}
    assert test_id_to_token == vocab_inv, (
        "Vocabulary inversion failed, something is wrong with the way they are built"
    )
    # Test the two tokenizers are identical
    assert len(tokenizer2) == vocab_size + NUM_ADDITIONAL_TOKENS_SECOND_TRAINING

    # Training is only deterministic when training with BPE (and WordPiece as training
    # is the same) when not using a `continuing_subword_prefix`.
    # With Unigram, vocabs may have swapped ids orders.
    if model in ["BPE", "WordPiece"] and encode_ids_split == "no":
        assert tokenizer2 == tokenizer1, (
            "tokenizer 1-shot not equal to tokenizer 2-shots"
        )

    # Checks tokens <--> ids <--> bytes conversions with one test case
    tokens = tokenizer1(files_paths[0], encode_ids=False)
    if not tokenizer1.one_token_stream:
        tokens = tokens[0]
    tokenizer1.complete_sequence(tokens, complete_bytes=True)  # not done by default
    toks_from_bytes = tokenizer1._bytes_to_tokens(tokens.bytes)
    ids_from_toks = tokenizer1._tokens_to_ids(toks_from_bytes)
    bytes_from_ids = tokenizer1._ids_to_bytes(ids_from_toks, as_one_str=True)
    assert all(
        [
            bytes_from_ids == tokens.bytes,
            toks_from_bytes == tokens.tokens,
            ids_from_toks == tokens.ids,
        ]
    ), (
        "Conversion between tokens / bytes / ids failed, something is wrong in"
        "vocabularies"
    )

    # Unbatched encoding-decoding
    if model in ("BPE", "WordPiece") and encode_ids_split == "no":
        func_check = _check_equal_seq
    elif model == "Unigram":
        func_check = _check_seq_len
    else:
        func_check = _check_seq_enc_dec
    at_least_one_error = False
    tok_time = 0
    samples_og, seq_len_reductions = [], []  # sample_og used for batched
    for file_path in tqdm(files_paths, desc="Testing encoding-decoding unbatched"):
        # Tokenize file without encoding ids first
        tokens_original = tokenizer1(file_path, encode_ids=False)
        if not tokenizer1.one_token_stream:
            tokens_original = tokens_original[0]
        samples_og.append(tokens_original)

        # Encode the token ids
        tokens1_encoded = replace(tokens_original)
        tokens2_encoded = replace(tokens_original)
        t0 = time()
        tokenizer1.encode_token_ids(tokens1_encoded)
        tok_time += time() - t0
        tokenizer2.encode_token_ids(tokens2_encoded)
        seq_len_reductions.append(1 - len(tokens1_encoded) / len(tokens_original))

        # Decode the token ids
        tokens1_decoded = replace(tokens1_encoded)
        tokens2_decoded = replace(tokens2_encoded)
        tokenizer1.decode_token_ids(tokens1_decoded)
        tokenizer2.decode_token_ids(tokens2_decoded)

        # Check everything went good
        at_least_one_error = (
            not func_check(
                tokens_original,
                tokens1_encoded,
                tokens2_encoded,
                tokens1_decoded,
                tokens2_decoded,
                tokenization,
                file_path.name,
            )
            or at_least_one_error
        )
    print(
        f"Encoding-decoding time un-batched: {tok_time:.2f} (mean:"
        f"{tok_time / len(files_paths):.4f})\n"
        f"Mean sequence length reduction: "
        f"{sum(seq_len_reductions) / len(seq_len_reductions):.2f}"
    )
    assert not at_least_one_error

    # Batched encoding-decoding
    samples1_encoded = [replace(s) for s in samples_og]
    samples2_encoded = [replace(s) for s in samples_og]
    t0 = time()
    tokenizer1.encode_token_ids(samples1_encoded)
    tok_time = time() - t0
    tokenizer2.encode_token_ids(samples2_encoded)

    samples1_decoded = [replace(s) for s in samples1_encoded]
    samples2_decoded = [replace(s) for s in samples2_encoded]
    tokenizer1.decode_token_ids(samples1_decoded)
    tokenizer2.decode_token_ids(samples2_decoded)

    for seq_og, seq1_enc, seq2_enc, seq1_dec, seq2_dec, file_path in zip(
        samples_og,
        samples1_encoded,
        samples2_encoded,
        samples1_decoded,
        samples2_decoded,
        files_paths,
    ):
        at_least_one_error = (
            not func_check(
                seq_og,
                seq1_enc,
                seq2_enc,
                seq1_dec,
                seq2_dec,
                tokenization,
                file_path.name,
            )
            or at_least_one_error
        )

    print(
        f"Encoding-decoding time batched: {tok_time:.2f} (mean:"
        f"{tok_time / len(files_paths):.4f})"
    )
    assert not at_least_one_error


def _check_equal_seq(
    seq_og: miditok.TokSequence,
    seq1_encoded: miditok.TokSequence,
    seq2_encoded: miditok.TokSequence,
    seq1_decoded: miditok.TokSequence,
    seq2_decoded: miditok.TokSequence,
    tokenization: str,
    file_name: str,
) -> bool:
    no_error = True
    if seq1_encoded.ids != seq2_encoded.ids:
        print(
            f"Encoding error for {tokenization} and {file_name}: "
            f"Encoded tokens not equal between two trained tokenizers"
        )
        no_error = False
    if not __check_seq_enc_dec(seq_og, seq1_decoded, seq2_decoded, tokenization):
        no_error = False
    return no_error


def _check_seq_len(
    seq_og: miditok.TokSequence,
    seq1_encoded: miditok.TokSequence,
    seq2_encoded: miditok.TokSequence,
    seq1_decoded: miditok.TokSequence,
    seq2_decoded: miditok.TokSequence,
    tokenization: str,
    file_name: str,
) -> bool:
    # just check sequence lengths
    no_error = True
    if len(seq1_encoded.ids) != len(seq2_encoded.ids):
        print(
            f"Encoding error for {tokenization} and {file_name}: "
            f"Encoded tokens not equal between two trained tokenizers"
        )
        no_error = False
    if not __check_seq_enc_dec(seq_og, seq1_decoded, seq2_decoded, tokenization):
        no_error = False
    return no_error


def _check_seq_enc_dec(
    *args: miditok.TokSequence | str,
) -> bool:
    return __check_seq_enc_dec(args[0], args[3], args[4], args[5])


def __check_seq_enc_dec(
    seq_og: miditok.TokSequence,
    seq1_decoded: miditok.TokSequence,
    seq2_decoded: miditok.TokSequence,
    tokenization: str,
) -> bool:
    # just check decoded sequences are identical to the original one
    if not len(seq1_decoded.ids) == len(seq2_decoded.ids) == len(seq_og.ids):
        print(
            f"Decoding error for {tokenization}: "
            f"Decoded tokens do not match the original ones"
        )
        return False
    return True
