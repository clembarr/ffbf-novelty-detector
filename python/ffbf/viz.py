from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_novelty_scores(
    scores: list[float],
    phases: list[tuple[int, str]] | None =None,
    ax: Axes | None =None
) -> Axes:
    """Line plot of novelty scores over time.
    Parameters:
        scores (list[float]): novelty score at each time step
        phases (list[tuple[int, str]] | None): vertical markers as (step_index, label) pairs
        ax (Axes | None): matplotlib axes to draw on; created if None
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(scores)
    ax.set_xlabel("Step")
    ax.set_ylabel("Novelty score")
    ax.set_title("Novelty scores over time")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    if phases:
        for idx, label in phases:
            ax.axvline(x=idx, color="gray", linestyle="--", linewidth=1)
            ax.text(idx + 0.2, 0.97, label, fontsize=8, color="gray", va="top")
    return ax


def plot_weight_state(
    weights: np.ndarray,
    title: str | None =None,
    ax: Axes | None =None
) -> Axes:
    """Bar chart of KC synaptic weights at a point in time.
    Parameters:
        weights (np.ndarray): 1D float32 array of KC weights, one bar per KC
        title (str | None): axes title; defaults to "Synaptic weight state"
        ax (Axes | None): matplotlib axes; created if None
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(np.arange(len(weights)), weights, width=1.0, linewidth=0)
    ax.set_xlabel("KC index")
    ax.set_ylabel("Weight")
    ax.set_title(title if title is not None else "Synaptic weight state")
    ax.set_xlim(-0.5, len(weights) - 0.5)
    ax.set_ylim(0, 1.05)
    return ax


def plot_weight_evolution(
    weight_history: np.ndarray,
    indices: list[int] | None =None,
    ax: Axes | None =None
) -> Axes:
    """Line plot of selected KC weights over time.
    Parameters:
        weight_history (np.ndarray): shape (n_steps, m) — weights captured at each step
        indices (list[int] | None): KC indices to plot; all KCs if None
        ax (Axes | None): matplotlib axes; created if None
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    if indices is None:
        indices = list(range(weight_history.shape[1]))
    for i in indices:
        ax.plot(weight_history[:, i], label=f"KC {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Weight")
    ax.set_title("Weight evolution")
    if len(indices) <= 10:
        ax.legend()
    return ax
