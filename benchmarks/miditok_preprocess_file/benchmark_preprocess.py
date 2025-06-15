#!/usr/bin/python3 python

"""Measure the average MIDI preprocessing speed."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from time import time

import numpy as np
from pandas import DataFrame, read_csv
from symusic import Score
from tqdm import tqdm

import miditok
from benchmarks.utils import mean_std_str
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


def benchmark_preprocess() -> None:
    r"""Read MIDI files and call `tokenizer.preprocess_score` on them."""
    results_path = HERE / "preprocess.csv"
    if results_path.is_file():
        df = read_csv(results_path, index_col=0)
    else:
        columns = ["symusic version"] + [
            f"{dataset} - {tokenization}"
            for dataset in DATASETS
            for tokenization in TOKENIZATIONS
        ]
        df = DataFrame(index=[], columns=columns)

    # Add a row to the dataframe
    index_name = f"miditok {version('miditok')}"
    df.at[index_name, "symusic version"] = version("symusic")

    for dataset in DATASETS:
        files_paths = list(
            (HERE.parent.parent.parent / "data" / dataset).rglob("*.mid")
        )[:MAX_NUM_FILES]
        for tokenization in TOKENIZATIONS:
            col_name = f"{dataset} - {tokenization}"
            tok_config = miditok.TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)
            tokenizer = getattr(miditok, tokenization)(tok_config)

            times = []
            for midi_path in tqdm(files_paths):
                try:
                    midi = Score(midi_path)
                except SCORE_LOADING_EXCEPTION:
                    continue
                t0 = time()
                tokenizer.preprocess_score(midi)
                times.append(time() - t0)

            times = np.array(times) * 1e3
            df.at[index_name, col_name] = f"{mean_std_str(times, 2)} ms"

    df.to_csv(HERE / "preprocess.csv")
    df.to_markdown(HERE / "preprocess.md")
    df.to_latex(HERE / "preprocess.txt")


if __name__ == "__main__":
    benchmark_preprocess()
