"""Utils methods for benchmarks."""

from __future__ import annotations

import numpy as np


def mean_std_str(
    dist: np.array | list[int | float], num_dec: int = 2, latex_pm: bool = False
) -> str:
    r"""
    Create a nice looking mean and standard deviation string of a distribution.

    :param dist: distribution to measure.
    :param num_dec: number of decimals to keep. (default: ``2``)
    :param latex_pm: whether to represent the "±" symbol with LaTeX command ("$\pm$").
        (default: ``False``)
    :return: string of the average and standard deviation of the distribution.
    """
    if not isinstance(dist, np.ndarray):
        dist = np.array(dist)
    mean, std = float(np.mean(dist)), float(np.std(dist))
    if latex_pm:
        return f"{mean:.{num_dec}f}" r"$\pm$" f"{std:.{num_dec}f}"  # noqa: ISC001
    return f"{mean:.{num_dec}f}±{std:.{num_dec}f}"
