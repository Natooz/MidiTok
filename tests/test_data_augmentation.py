"""Test methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from symusic import Score
from tqdm import tqdm

from miditok.data_augmentation import (
    augment_dataset,
)

from .utils_tests import HERE

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("data_path", [HERE / "MIDIs_multitrack"])
def test_augment_dataset(tmp_path: Path, data_path: Path) -> None:
    # We only test data augmentation on MIDIs with one tokenization

    midi_aug_path = tmp_path / "Multitrack_MIDIs_aug"
    min_duration = 0.03125
    augment_dataset(
        data_path,
        pitch_offsets=[-2, 1, 2],
        velocity_offsets=[-4, 5],
        duration_offsets=[-0.5, 1],
        all_offset_combinations=True,
        min_duration=min_duration,
        out_path=midi_aug_path,
    )

    aug_midi_paths = list(midi_aug_path.glob("**/*.mid"))
    for aug_midi_path in tqdm(
        aug_midi_paths, desc="CHECKING DATA AUGMENTATION ON MIDIS"
    ):
        # Determine offsets of file
        parts = aug_midi_path.stem.split("#")
        # If original non-augmented file
        if len(parts) < 2:
            continue
        original_stem, offsets_str = parts[0], parts[1].split("_")
        offsets = [0, 0, 0]
        for offset_str in offsets_str:
            for pos, letter in enumerate(["p", "v", "d"]):
                if offset_str[0] == letter:
                    offsets[pos] = int(offset_str[1:])

        # Loads MIDIs to compare
        midi_aug = Score(aug_midi_path)
        midi_ogi = Score(data_path / f"{original_stem}.mid")
        min_duration_ticks = round(min_duration * midi_aug.ticks_per_quarter)

        # Compare them
        for track_ogi, track_aug in zip(midi_ogi.tracks, midi_aug.tracks):
            if track_ogi.is_drum:
                continue
            track_ogi.notes.sort(key=lambda x: (x.start, x.pitch, x.end, x.velocity))
            track_aug.notes.sort(key=lambda x: (x.start, x.pitch, x.end, x.velocity))
            for note_o, note_a in zip(track_ogi.notes, track_aug.notes):
                if note_a.pitch != note_o.pitch + offsets[0]:
                    msg = (
                        f"Pitch assertion failed: expected "
                        f"{note_o.pitch + offsets[0]}, got {note_a.pitch}"
                    )
                    raise ValueError(msg)
                # If negative duration offset, dur_exp must be greater or equal than
                # min_duration_ticks
                if offsets[2] < 0:
                    dur_exp = max(
                        note_o.duration + offsets[2],
                        min_duration_ticks,
                    )
                # If positive duration offset, the original duration was just shifted
                elif offsets[2] > 0:
                    dur_exp = note_o.duration + offsets[2]
                else:
                    dur_exp = note_o.duration
                if note_a.duration != dur_exp:
                    msg = (
                        f"Duration assertion failed: expected {dur_exp}, got "
                        f"{note_a.duration}"
                    )
                    raise ValueError(msg)
            # We need to resort the tracks with the velocity key in third position
            # before checking their values.
            track_ogi.notes.sort(key=lambda x: (x.start, x.pitch, x.velocity))
            track_aug.notes.sort(key=lambda x: (x.start, x.pitch, x.velocity))
            for note_o, note_a in zip(track_ogi.notes, track_aug.notes):
                if note_a.velocity not in [1, 127, note_o.velocity + offsets[1]]:
                    msg = (
                        f"Velocity assertion failed: expected one in "
                        f"{[1, 127, note_o.velocity + offsets[1]]}, got {note_a.pitch}"
                    )
                    raise ValueError(msg)
