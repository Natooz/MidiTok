"""Test methods."""

from collections.abc import Callable
from pathlib import Path

import pytest

from miditok import TSD, TokenizerConfig, TokSequence

from .utils_tests import MIDI_PATHS_MULTITRACK


def test_tokseq_concat():
    ids1 = list(range(10))
    ids2 = list(range(10, 20))
    str1 = [str(id_ * 2) for id_ in ids1]
    str2 = [str(id_ * 2) for id_ in ids2]
    bytes1 = "".join(str1)
    bytes2 = "".join(str2)

    tokseq1 = TokSequence(ids=ids1, tokens=str1, bytes=bytes1)
    tokseq2 = TokSequence(ids=ids2, tokens=str2, bytes=bytes2)
    seq_concat = tokseq1 + tokseq2

    assert seq_concat.ids == ids1 + ids2
    assert seq_concat.tokens == str1 + str2
    assert seq_concat.bytes == bytes1 + bytes2


def test_tokseq_slice_and_concat():
    ids1 = list(range(20))
    str1 = [str(id_ * 2) for id_ in ids1]
    bytes1 = "".join(str1)

    tokseq = TokSequence(ids=ids1, tokens=str1, bytes=bytes1)
    subseq1 = tokseq[:10]
    subseq2 = tokseq[10:]

    assert subseq1.ids == ids1[:10]
    assert subseq1.tokens == str1[:10]
    assert subseq1.bytes == bytes1[:10]
    assert subseq2.ids == ids1[10:]
    assert subseq2.tokens == str1[10:]
    assert subseq2.bytes == bytes1[10:]

    tokseq_concat = subseq1 + subseq2
    assert tokseq == tokseq_concat


@pytest.mark.parametrize("file_path", MIDI_PATHS_MULTITRACK, ids=lambda p: p.name)
@pytest.mark.parametrize("tokenization", [TSD], ids=lambda c: c.__name__)
def test_split_tokseq_per_bars_beats(file_path: Path, tokenization: Callable):
    tokenizer = tokenization(TokenizerConfig(use_programs=True))
    tokseq = tokenizer(file_path)

    # Split per bars
    seqs = tokseq.split_per_bars()
    concat_seq = seqs.pop(0)
    for seq in seqs:
        concat_seq += seq
    assert concat_seq == tokseq

    # Split per beats
    seqs = tokseq.split_per_beats()
    concat_seq = seqs.pop(0)
    for seq in seqs:
        concat_seq += seq
    assert concat_seq == tokseq
