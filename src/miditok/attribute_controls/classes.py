"""Common classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import sample, uniform
from typing import TYPE_CHECKING

import numpy as np

from miditok.utils import get_bars_ticks

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from symusic import Score
    from symusic.core import TrackTick

    from miditok import Event


class AttributeControl(ABC):
    r"""
    Attribute Control class, defining the possible values and their computation.

    **Note:** track-level attribute controls need to return :class:`miditok.Event`\s
    with ``time`` attributes set to -1, as they need to be at the very first positions
    in the token sequence after sorting the list of events. Their times will be set to
    0 after sorting in ``MusicTokenizer._score_to_tokens``.

    :param tokens: tokens of the attribute control as a list of tuples specifying
        their types and values.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = tokens

    @abstractmethod
    def compute(
        self,
        track: TrackTick,
        time_division: int,
        ticks_bars: Sequence[int],
        ticks_beats: Sequence[int],
        bars_idx: Sequence[int],
    ) -> list[Event]:
        """
        Compute the attribute control from a ``symusic.Track``.

        :param track: ``symusic.Track`` object to compute the attribute from.
        :param time_division: time division in ticks per quarter note of the file.
        :param ticks_bars: ticks indicating the beginning of each bar.
        :param ticks_beats: ticks indicating the beginning of each beat.
        :param bars_idx: **sorted** indexes of the bars to compute the bar-level control
            attributes from. If ``None`` is provided, the attribute controls are
            computed on all the bars. (default: ``None``)
        :return: attribute control values.
        """
        raise NotImplementedError


class BarAttributeControl(AttributeControl, ABC):
    """Base class for bar-level attribute controls."""

    def compute(
        self,
        track: TrackTick,
        time_division: int,
        ticks_bars: Sequence[int],
        ticks_beats: Sequence[int],
        bars_idx: Sequence[int],
    ) -> list[Event]:
        """
        Compute the attribute control from a ``symusic.Track``.

        :param track: ``symusic.Track`` object to compute the attribute from.
        :param time_division: time division in ticks per quarter note of the file.
        :param ticks_bars: ticks indicating the beginning of each bar.
        :param ticks_beats: ticks indicating the beginning of each beat.
        :param bars_idx: **sorted** indexes of the bars to compute the bar-level control
            attributes from. If ``None`` is provided, the attribute controls are
            computed on all the bars. (default: ``None``)
        :return: attribute control values.
        """
        del ticks_beats
        # List with indices of non-empty bars of the shape:
        # [track, bar, (bar_tick, [bar_attributes])]
        attribute_controls = []
        ticks_bars = np.array(ticks_bars)

        # Iterate over each track
        notes_soa = track.notes.numpy()
        controls_soa = track.controls.numpy()
        pitch_bends_soa = track.pitch_bends.numpy()
        bar_ticks_track = ticks_bars[np.where(ticks_bars <= notes_soa["time"][-1])[0]]

        # Iterate over each bar
        # The bars idx should be sorted, so we can store note idx to speed up the
        # notes search process with np.where
        note_start_idx = control_start_idx = pitch_bend_start_idx = 0
        for bar_idx in bars_idx:
            # Skip this bar if it is beyond the content of the current track
            if bar_idx >= len(bar_ticks_track):
                continue
            # This is the last bar, idx of end note is `None`
            if bar_idx == len(bar_ticks_track) - 1:
                note_end_idx = control_end_idx = pitch_bend_end_idx = None  # last bar
            # Get the idx of the last note of the current bar
            else:
                idx_notes = np.where(
                    notes_soa["time"][note_start_idx:] >= bar_ticks_track[bar_idx + 1]
                )[0]
                # if len == 0, there is no notes after the current bar
                note_end_idx = (
                    None if len(idx_notes) == 0 else idx_notes[0] + note_start_idx
                )
                idx_controls = np.where(
                    controls_soa["time"][control_start_idx:]
                    >= bar_ticks_track[bar_idx + 1]
                )[0]
                control_end_idx = (
                    None
                    if len(idx_controls) == 0
                    else idx_controls[0] + control_start_idx
                )
                idx_pitch_bends = np.where(
                    pitch_bends_soa["time"][pitch_bend_start_idx:]
                    >= bar_ticks_track[bar_idx + 1]
                )[0]
                pitch_bend_end_idx = (
                    None
                    if len(idx_pitch_bends) == 0
                    else idx_pitch_bends[0] + pitch_bend_start_idx
                )

            # Compute attribute if the bar is not empty
            if note_end_idx is None or (
                note_end_idx and note_end_idx > note_start_idx + 1
            ):
                notes_soa_bar = {
                    key: value[note_start_idx:note_end_idx]
                    for key, value in notes_soa.items()
                }
                controls_soa_bar = {
                    key: value[control_start_idx:control_end_idx]
                    for key, value in controls_soa.items()
                }
                pitch_bends_soa_bar = {
                    key: value[pitch_bend_start_idx:pitch_bend_end_idx]
                    for key, value in pitch_bends_soa.items()
                }
                # Check a second time in case it is the last bar
                if len(notes_soa_bar["time"]) > 0:
                    attribute_controls_bar = self._compute_on_bar(
                        notes_soa_bar,
                        controls_soa_bar,
                        pitch_bends_soa_bar,
                        time_division,
                    )
                    # Set time to events
                    for event in attribute_controls_bar:
                        event.time = ticks_bars[bar_idx]
                    attribute_controls += attribute_controls_bar

            # Break the loop early if the last note is reached
            if note_end_idx is None:
                break
            # for next iteration
            note_start_idx = note_end_idx
            control_start_idx = control_end_idx
            pitch_bend_start_idx = pitch_bend_end_idx

        return attribute_controls

    @abstractmethod
    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        """
        Compute the attribute controls from a specific bar.

        :param notes_soa: structure of arrays of notes.
        :param controls_soa: structure of arrays of control changes.
        :param pitch_bends_soa: structure of arrays of pitch bends.
        :param time_division: time division in ticks per quarter note of the file.
        """
        raise NotImplementedError


def create_random_ac_indexes(
    score: Score,
    attribute_controls: Sequence[AttributeControl],
    tracks_idx_ratio: float | tuple[float, float] | None = None,
    bars_idx_ratio: float | tuple[float, float] | None = None,
) -> Mapping[int, Mapping[int, bool | Sequence[int]]]:
    """
    Randomly create tracks and bars indexes for attribute controls computation.

    :param score: ``symusic.Score`` to set the indexes for.
    :param attribute_controls: attribute controls that will be computed. They need to be
        provided to get their indexes.
    :param tracks_idx_ratio: ratio or range of ratio (between 0 and 1) of track-level
        attribute controls per track. (default ``None``)
    :param bars_idx_ratio: ratio or range of ratio (between 0 and 1) of track-level
        attribute controls per track. (default ``None``)
    :return: indexes of attribute controls to be used when tokenizing a music file.
    """
    acs_track_idx, acs_bars_idx = [], []
    for i in range(len(attribute_controls)):
        if isinstance(attribute_controls[i], BarAttributeControl):
            acs_bars_idx.append(i)
        else:
            acs_track_idx.append(i)
    bar_ticks = np.array(get_bars_ticks(score, only_notes_onsets=True))

    ac_indexes = {}
    for track_idx, track in enumerate(score.tracks):
        track_indexes = {}
        # Randomly select track_acs
        if tracks_idx_ratio:
            num_track_acs = round(
                len(acs_track_idx)
                * (
                    tracks_idx_ratio
                    if isinstance(tracks_idx_ratio, (float, int))
                    else uniform(*tracks_idx_ratio)  # noqa: S311
                )
            )
            track_indexes = dict.fromkeys(sample(acs_track_idx, k=num_track_acs), True)
        # For each "bar-level" ac randomly sample the bars to compute
        if bars_idx_ratio:
            bar_ticks_track = bar_ticks[np.where(bar_ticks <= track.end())[0]]
            for ac_idx in acs_bars_idx:
                num_bars = round(
                    len(bar_ticks_track)
                    * (
                        bars_idx_ratio
                        if isinstance(bars_idx_ratio, (float, int))
                        else uniform(*bars_idx_ratio)  # noqa: S311
                    )
                )
                track_indexes[ac_idx] = sorted(
                    sample(list(range(len(bar_ticks_track))), k=num_bars)
                )
        ac_indexes[track_idx] = track_indexes

    return ac_indexes
