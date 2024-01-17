#!/usr/bin/python3 python

"""Benchmark for Python MIDI parsing libraries."""

from __future__ import annotations

import random
from importlib.metadata import version
from pathlib import Path
from platform import processor, system
from time import time

import numpy as np
from miditoolkit import MidiFile
from pandas import DataFrame
from pretty_midi import PrettyMIDI
from symusic import Score
from tqdm import tqdm

HERE = Path(__file__).parent
DATASETS = ["Maestro", "MMD", "POP909"]
LIBRARIES = ["Symusic", "MidiToolkit", "Pretty MIDI"]
MAX_NUM_FILES = 500


def read_midi_files(
    midi_paths: list[Path]
) -> tuple[list[float], list[float], list[float]]:
    times_mtk, times_sms, times_ptm = [], [], []
    for midi_path in tqdm(midi_paths, desc="Loading MIDIs"):
        # Miditoolkit
        t0 = time()
        _ = MidiFile(midi_path)
        times_mtk.append(time() - t0)

        # Symusic
        t0 = time()
        _ = Score(midi_path)
        times_sms.append(time() - t0)

        # Pretty MIDI
        t0 = time()
        _ = PrettyMIDI(str(midi_path))
        times_ptm.append(time() - t0)
    return times_sms, times_mtk, times_ptm


def benchmark_midi_parsing(
    seed: int = 777,
):
    r"""
    Measure the reading time of MIDI files with different libraries.

    :param seed: random seed
    """
    random.seed(seed)
    print(f"CPU: {processor()}")
    print(f"System: {system()}")
    for library in ["symusic", "miditoolkit", "pretty_midi"]:
        print(f"{library}: {version(library)}")

    df = DataFrame(index=LIBRARIES, columns=DATASETS)

    # Record times
    for dataset in DATASETS:
        midi_paths = list((HERE.parent.parent / "data" / dataset).rglob("*.mid"))[
            :MAX_NUM_FILES
        ]
        all_times = read_midi_files(midi_paths)
        for library, times in zip(LIBRARIES, all_times):
            times_ = np.array(times)
            if library == "Symusic":
                times_ *= 1e3
                unit = "ms"
            else:
                unit = "sec"
            df.at[
                library, dataset
            ] = f"{np.mean(times_):.2f} ± {np.std(times_):.2f} {unit}"

    print(df)
    df.to_csv(HERE / "benchmark_midi_read.csv")
    df.to_markdown(HERE / "benchmark_midi_read.md")


if __name__ == "__main__":
    benchmark_midi_parsing()
