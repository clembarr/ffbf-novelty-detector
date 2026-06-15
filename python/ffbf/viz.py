from __future__ import annotations
import numpy as np


def plot_novelty_scores(
    scores: list[float],
    ax =None
):
    """Line plot of novelty scores over time.
    Parameters:
        scores (list[float]): novelty score at each time step
        ax (Axes): matplotlib axes to draw on; created if None
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(scores)
    ax.set_xlabel("Step")
    ax.set_ylabel("Novelty score")
    ax.set_title("Novelty scores over time")
    return ax


def plot_weight_distribution(
    weights: np.ndarray,
    ax =None
):
    """Histogram of synaptic weight values at a point in time.
    Parameters:
        weights (np.ndarray): 1D float32 array of synaptic weights
        ax (Axes): matplotlib axes; created if None
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(weights, bins=30)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Synaptic weight distribution")
    return ax


def plot_weight_evolution(
    weight_history: np.ndarray,
    indices: list[int] =None,
    ax =None
):
    """Line plot of selected KC weights over time.
    Parameters:
        weight_history (np.ndarray): shape (n_steps, m) — weights captured at each step
        indices (list[int]): KC indices to plot; all KCs if None
        ax (Axes): matplotlib axes; created if None
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
