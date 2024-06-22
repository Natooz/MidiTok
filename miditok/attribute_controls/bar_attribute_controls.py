"""Bar-level attribute controls modules."""

from __future__ import annotations

import numpy as np

from miditok import Event

from .classes import BarAttributeControl


class BarOnsetPolyphony(BarAttributeControl):
    """
    Onset polyphony attribute control at the bar level.

    It specifies the minimum and maximum number of notes played simultaneously at a
    given time onset.
    It can be enabled with the ``ac_polyphony_bar`` argument of
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
                for tok_type in ("ACBarOnsetPolyphonyMin", "ACBarOnsetPolyphonyMax")
                for val in range(polyphony_min, polyphony_max + 1)
            ],
        )

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa, time_division
        _, counts_onsets = np.unique(notes_soa["time"], return_counts=True)
        onset_poly_min, onset_poly_max = np.min(counts_onsets), np.max(counts_onsets)

        min_poly = min(max(onset_poly_min, self.min_polyphony), self.max_polyphony)
        max_poly = min(onset_poly_max, self.max_polyphony)
        return [
            Event("ACBarOnsetPolyphonyMin", min_poly),
            Event("ACBarOnsetPolyphonyMax", max_poly),
        ]


class BarPitchClass(BarAttributeControl):
    """
    Bar-level pitch classes attribute control.

    This attribute control specifies which pitch classes are present within a bar.
    """

    def __init__(self) -> None:
        super().__init__(tokens=[f"ACBarPitchClass_{i}" for i in range(12)])

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa, time_division
        pitch_values = notes_soa["pitch"] % 12
        pitch_values = np.unique(pitch_values)
        return [Event("ACBarPitchClass", pitch) for pitch in pitch_values]


class BarNoteDensity(BarAttributeControl):
    """
    Bar-level note density attribute control.

    It specifies the number of notes per bar. If a bar contains more that the maximum
    density (``density_max``), a ``density_max+`` token will be returned.

    :param density_max: maximum note density per bar to consider.
    """

    def __init__(self, density_max: int) -> None:
        self.density_max = density_max
        super().__init__(
            tokens=[
                *(f"ACBarNoteDensity_{i}" for i in range(density_max)),
                f"ACBarNoteDensity_{self.density_max}+",
            ],
        )

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa, time_division
        n_notes = len(notes_soa["time"])
        if n_notes >= self.density_max:
            return [Event("ACBarNoteDensity", f"{self.density_max}+")]
        return [Event("ACBarNoteDensity", n_notes)]


class BarNoteDuration(BarAttributeControl):
    """
    Note duration attribute control.

    This attribute controls specifies the note durations (whole, half, quarter, eight,
    sixteenth and thirty-second) present in a bar.
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
                f"ACBarNoteDuration{duration}_{val}"
                for duration in self._note_durations
                for val in (0, 1)
            ],
        )
        # Factors multiplying ticks/quarter time division
        self.factors = (4, 2, 1, 0.5, 0.25)

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa
        durations = np.unique(notes_soa["duration"])
        controls = []
        for fi, factor in enumerate(self.factors):
            controls.append(
                Event(
                    f"ACBarNoteDuration{self._note_durations[fi]}",
                    1 if time_division * factor in durations else 0,
                )
            )
        return controls
