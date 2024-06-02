#!/usr/bin/python3 python

"""Benchmark for Python MIDI parsing libraries."""

from __future__ import annotations

import random
from pathlib import Path
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
MAX_NUM_FILES = 1000


def read_midi_files(
    midi_paths: list[Path],
) -> tuple[list[float], list[float], list[float]]:
    """
    Read a list of MIDI files and return their reading times.

    :param midi_paths: paths to the midi files to read.
    :return: times of files reads for symusic, miditoolkit and pretty_midi.
    """
    times_mtk, times_sms, times_ptm = [], [], []
    for midi_path in tqdm(midi_paths, desc="Loading MIDIs"):
        # We count times only if all libraries load the file without error
        try:
            # Miditoolkit
            t0 = time()
            _ = MidiFile(midi_path)
            t_mtk = time() - t0

            # Symusic
            t0 = time()
            _ = Score(midi_path)
            t_sms = time() - t0

            # Pretty MIDI
            t0 = time()
            _ = PrettyMIDI(str(midi_path))
            t_ptm = time() - t0
        except:  # noqa: E722, S112
            continue

        times_mtk.append(t_mtk)
        times_sms.append(t_sms)
        times_ptm.append(t_ptm)

    return times_sms, times_mtk, times_ptm


def benchmark_midi_parsing(
    seed: int = 777,
) -> None:
    r"""
    Measure the reading time of MIDI files with different libraries.

    :param seed: random seed
    """
    random.seed(seed)

    df = DataFrame(index=LIBRARIES, columns=DATASETS)

    # Record times
    for dataset in DATASETS:
        midi_paths = list(
            (HERE.parent.parent.parent / "data" / dataset).rglob("*.mid")
        )[:MAX_NUM_FILES]
        all_times = read_midi_files(midi_paths)
        for library, times in zip(LIBRARIES, all_times):
            times_ = np.array(times)
            if library == "Symusic":
                times_ *= 1e3
                unit = "ms"
            else:
                unit = "sec"
            df.at[library, dataset] = (
                f"{np.mean(times_):.2f} ± {np.std(times_):.2f} {unit}"
            )

    df.to_csv(HERE / "midi_read.csv")
    df.to_markdown(HERE / "midi_read.md")
    df.to_latex(HERE / "midi_read.txt")


if __name__ == "__main__":
    benchmark_midi_parsing()
