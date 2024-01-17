#!/usr/bin/python3 python

"""Measure the average MIDI tokenization speed."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from platform import processor, system
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
MIDI_PATHS_ONE_TRACK = sorted((HERE / "MIDIs_one_track").rglob("*.mid"))
MIDI_PATHS_MULTITRACK = sorted((HERE / "MIDIs_multitrack").rglob("*.mid"))
MIDI_PATHS_ALL = sorted(
    deepcopy(MIDI_PATHS_ONE_TRACK) + deepcopy(MIDI_PATHS_MULTITRACK)
)
TOKENIZATIONS = ["REMI", "TSD", "MIDILike", "Structured"]


def benchmark_tokenize(midi_paths: list[Path]):
    r"""
    Read MIDI files and call `tokenizer.preprocess_midi` on them.

    :param midi_paths: list of paths to MIDI files.
    """
    print(f"CPU: {processor()}")
    print(f"System: {system()}")

    for tokenization in TOKENIZATIONS:
        tok_config = miditok.TokenizerConfig(**TOKENIZER_CONFIG_KWARGS)
        tokenizer = getattr(miditok, tokenization)(tok_config)
        times = []
        for midi_path in midi_paths:
            midi = Score(midi_path)
            t0 = time()
            tokenizer.midi_to_tokens(midi)
            t1 = time() - t0
            times.append(t1)

        times = np.array(times)
        mean = np.mean(times)
        std = np.std(times)
        print(f"{tokenization}: {mean:.3f} ± {std:.3f} sec")


if __name__ == "__main__":
    benchmark_tokenize(MIDI_PATHS_ALL)
