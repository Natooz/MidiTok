"""Track-level attribute controls modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miditok import Event

from .classes import AttributeControl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from symusic.core import TrackTick


class TrackOnsetPolyphony(AttributeControl):
    """
    Onset polyphony attribute control at the track level.

    It specifies the minimum and maximum number of notes played simultaneously at a
    given time onset.
    It can be enabled with the ``ac_polyphony_track`` argument of
    :class:`miditok.TokenizerConfig`.

    :param polyphony_min: minimum number of simultaneous notes to consider.
    :param polyphony_max: maximum number of simultaneous notes to consider.
    """

    def __init__(
        self,
        polyphony_min: int,
        polyphony_max: int,
    ) -> None:
        self.min_polyphony = polyphony_min
        self.max_polyphony = polyphony_max
        super().__init__(
            tokens=[
                f"{tok_type}_{val}"
                for tok_type in ("ACTrackOnsetPolyphonyMin", "ACTrackOnsetPolyphonyMax")
                for val in range(polyphony_min, polyphony_max + 1)
            ],
        )

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
        del time_division, ticks_bars, ticks_beats, bars_idx
        notes_soa = track.notes.numpy()
        unique_onsets, counts_onsets = np.unique(notes_soa["time"], return_counts=True)
        onset_poly_min, onset_poly_max = np.min(counts_onsets), np.max(counts_onsets)
        if onset_poly_min > self.min_polyphony:
            onset_poly_min = self.min_polyphony
        return [
            Event(
                "ACTrackOnsetPolyphonyMin",
                max(onset_poly_min, self.min_polyphony),
                -1,
            ),
            Event(
                "ACTrackOnsetPolyphonyMax",
                min(onset_poly_max, self.max_polyphony),
                -1,
            ),
        ]


class TrackNoteDuration(AttributeControl):
    """
    Note duration attribute control.

    This attribute controls specifies the note durations (whole, half, quarter, eight,
    sixteenth and thirty-second) present in a track.
    """

    def __init__(self) -> None:
        self._note_durations = (
            "Whole",
            "Half",
            "Quarter",
            "Eight",
            "Sixteenth",
            "ThirtySecond",
        )
        super().__init__(
            tokens=[
                f"ACTrackNoteDuration{duration}_{val}"
                for duration in self._note_durations
                for val in (0, 1)
            ],
        )
        # Factors multiplying ticks/quarter time division
        self.factors = (4, 2, 1, 0.5, 0.25)

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
        del ticks_bars, ticks_beats, bars_idx
        durations = np.unique(track.notes.numpy()["duration"])
        controls = []
        for fi, factor in enumerate(self.factors):
            controls.append(
                Event(
                    f"ACTrackNoteDuration{self._note_durations[fi]}",
                    1 if time_division * factor in durations else 0,
                    -1,
                )
            )
        return controls


class TrackNoteDensity(AttributeControl):
    """
    Track-level note density attribute control.

    It specifies the minimum and maximum number of notes per bar within a track.
    If a bar contains more that the maximum density (``density_max``), a
    ``density_max+`` token will be returned.

    :param density_min: minimum note density per bar to consider.
    :param density_max: maximum note density per bar to consider.
    """

    def __init__(self, density_min: int, density_max: int) -> None:
        self.density_min = density_min
        self.density_max = density_max
        super().__init__(
            tokens=[
                *(
                    f"ACTrackNoteDensityMin_{i}"
                    for i in range(density_min, density_max)
                ),
                *(
                    f"ACTrackNoteDensityMax_{i}"
                    for i in range(density_min, density_max)
                ),
                f"ACTrackNoteDensityMin_{density_max}+",
                f"ACTrackNoteDensityMax_{density_max}+",
            ],
        )

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
        del time_division, ticks_beats, bars_idx
        notes_soa = track.notes.numpy()
        notes_ticks = notes_soa["time"]

        ticks_bars = (
            ticks_bars.copy() if isinstance(ticks_bars, list) else list(ticks_bars)
        )
        if (track_end_tick := track.end()) > ticks_bars[-1]:
            ticks_bars.append(track_end_tick + 1)
        bar_note_density, _ = np.histogram(notes_ticks, bins=ticks_bars)
        bar_density_min = np.min(bar_note_density)
        bar_density_max = np.max(bar_note_density)

        controls = []

        if bar_density_min >= self.density_max:
            controls.append(Event("ACTrackNoteDensityMin", f"{self.density_max}+", -1))
            controls.append(Event("ACTrackNoteDensityMax", f"{self.density_max}+", -1))
        else:
            controls.append(Event("ACTrackNoteDensityMin", bar_density_min, -1))
            if bar_density_max >= self.density_max:
                controls.append(
                    Event("ACTrackNoteDensityMax", f"{self.density_max}+", -1)
                )
            else:
                controls.append(Event("ACTrackNoteDensityMax", bar_density_max, -1))

        return controls


class TrackRepetition(AttributeControl):
    """
    Track-level repetition level between consecutive bars.

    This attribute corresponds to the average similarity between consecutive bars,
    with the similarity between too bars computed as the ratio of "logical and"
    positions between their binary pianoroll matrices.
    For each bar, the module will compute its similarity with the next
    ``num_consecutive_bars`` bars, and return the average of all the similarities.

    :param num_bins: number of levels of repetitions.
    :param num_consecutive_bars: number of successive bars to compare the similarity
        with each current bar.
    :param pitch_range: pitch range of the tokenizer.
    """

    def __init__(
        self, num_bins: int, num_consecutive_bars: int, pitch_range: tuple[int, int]
    ) -> None:
        self.num_bins = num_bins
        self.num_consecutive_bars = num_consecutive_bars
        self._pitch_range = pitch_range
        self._bins = np.linspace(0, 1, num_bins)
        super().__init__(
            tokens=[f"ACTrackRepetition_{i:.2f}" for i in self._bins],
        )

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
        del time_division, ticks_beats, bars_idx
        pianoroll = track.pianoroll(
            ["onset", "offset"],  # only comparing onsets and offsets for better acc
            self._pitch_range,
            False,  # noqa: FBT003
        ).transpose(2, 1, 0)  # (2,P,T) --> (T,P,2)

        similarities = []
        for bar_idx, bar_tick in enumerate(ticks_bars[:-1]):
            if bar_tick > pianoroll.shape[0]:
                break  # track ended at previous bar
            bar1 = pianoroll[bar_tick : ticks_bars[bar_idx + 1]]
            if (num_assertions := np.count_nonzero(bar1)) == 0:
                continue
            # Iterate over next bars to measure similarities
            for bar2_idx in range(
                bar_idx + 1,
                min(len(ticks_bars), bar_idx + self.num_consecutive_bars),
            ):
                bar2 = pianoroll[
                    ticks_bars[bar2_idx] : ticks_bars[bar2_idx + 1]
                    if bar2_idx + 1 < len(ticks_bars)
                    else None
                ]
                if bar1.shape == bar2.shape:
                    similarities.append(np.count_nonzero(bar1 & bar2) / num_assertions)

        if len(similarities) > 0:
            idx = (np.abs(self._bins - np.mean(np.array(similarities)))).argmin()
            return [Event("ACTrackRepetition", f"{self._bins[idx]:.2f}", -1)]
        return []
