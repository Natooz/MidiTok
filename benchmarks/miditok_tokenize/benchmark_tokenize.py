#!/usr/bin/python3 python

"""Measure the average MIDI tokenization speed."""

from __future__ import annotations

from pathlib import Path
from time import time

import numpy as np
from pandas import DataFrame, read_csv
from symusic import Score
from tqdm import tqdm

import miditok
from benchmarks import mean_std_str
from miditok.constants import SCORE_LOADING_EXCEPTION

TOKENIZER_CONFIG_KWARGS = {
    "use_tempos": True,
    "use_time_signatures": True,
    "use_sustain_pedals": True,
    "use_pitch_bends": True,
    "log_tempos": True,
    "beat_res": {(0, 4): 8, (4, 12): 4, (12, 16): 2},
    "delete_equal_successive_time_sig_changes": True,
    "delete_equal_successive_tempo_changes": True,
}

HERE = Path(__file__).parent
TOKENIZATIONS = ["REMI", "TSD", "MIDILike", "Structured"]
DATASETS = ["Maestro", "MMD", "POP909"]
MAX_NUM_FILES = 1000


def benchmark_tokenize() -> None:
    r"""Read MIDI files and tokenize them."""
    results_path = HERE / "tokenize.csv"
    if results_path.is_file():
        df = read_csv(results_path, index_col=0)
    else:
        df = DataFrame(index=TOKENIZATIONS, columns=DATASETS)

    for dataset in DATASETS:
        midi_paths = list(
            (HERE.parent.parent.parent / "data" / dataset).rglob("*.mid")
        )[:MAX_NUM_FILES]
        for tokenization in TOKENIZATIONS:
            tok_config = miditok.TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)
            tokenizer = getattr(miditok, tokenization)(tok_config)

            times = []
            for midi_path in tqdm(midi_paths):
                try:
                    midi = Score(midi_path)
                except SCORE_LOADING_EXCEPTION:
                    continue
                t0 = time()
                tokenizer.encode(midi)
                times.append(time() - t0)

            times = np.array(times) * 1e3
            df.at[tokenization, dataset] = f"{mean_std_str(times, 2)} ms"

    df.to_csv(HERE / "tokenize.csv")
    df.to_markdown(HERE / "tokenize.md")
    df.to_latex(HERE / "tokenize.txt")


if __name__ == "__main__":
    benchmark_tokenize()
