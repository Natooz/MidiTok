#!/usr/bin/python3 python

"""Measure the average MIDI preprocessing speed."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path
from time import time

import numpy as np
from symusic import Score

import miditok

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
    r"""Read MIDI files and call `tokenizer.preprocess_midi` on them."""
    results = []
    for dataset in DATASETS:
        midi_paths = list(
            (HERE.parent.parent.parent / "data" / dataset).rglob("*.mid")
        )[:MAX_NUM_FILES]

        for tokenization in TOKENIZATIONS:
            tok_config = miditok.TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)
            tokenizer = getattr(miditok, tokenization)(tok_config)

            times = []
            for midi_path in midi_paths:
                try:
                    midi = Score(midi_path)
                    # midi = MidiFile(midi_path)
                    t0 = time()
                    tokenizer.preprocess_midi(midi)
                    times.append(time() - t0)
                except:  # noqa: E722, S112
                    continue

            times = np.array(times) * 1e3
            mean = np.mean(times)
            std = np.std(times)
            results.append(f"{mean:.2f} ± {std:.2f} ms")

    csv_file_path = HERE / "benchmark_preprocess.csv"
    write_header = not csv_file_path.is_file()
    with Path.open(csv_file_path, "a") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            header = ["MidiTok version"] + [
                f"{dataset} - {tokenization}"
                for dataset in DATASETS
                for tokenization in TOKENIZATIONS
            ]
            writer.writerow(header)
        writer.writerow([version("miditok"), *results])


if __name__ == "__main__":
    benchmark_preprocess()
