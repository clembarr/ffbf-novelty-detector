from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D


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


def _pca(X: np.ndarray, n_components: int =3) -> np.ndarray:
    #project X onto its top n_components principal axes via thin SVD
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


def plot_embedding_3d(
    embeddings: np.ndarray,
    labels: list[str],
    novelty_scores: list[float],
    ax: Axes3D | None =None
) -> Axes3D:
    """3D scatter of embeddings projected to 3 PCA components.
    Parameters:
        embeddings (np.ndarray): shape (n, d) float32 embedding matrix
        labels (list[str]): domain label per point, used for color grouping
        novelty_scores (list[float]): novelty score per point in [0, 1], controls marker size
        ax (Axes3D | None): 3D axes; created if None
    Returns:
        Axes3D: the 3D axes used for plotting
    Raises:
        ValueError: if embeddings has fewer than 3 columns
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401 — registers "3d" projection
    if embeddings.shape[1] < 3:
        raise ValueError(f"embeddings must have at least 3 columns, got {embeddings.shape[1]}")
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    proj = _pca(embeddings)
    unique_labels = sorted(set(labels))
    cmap = plt.cm.tab10
    colors = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}
    sizes = np.array(novelty_scores) * 80 + 20
    label_arr = np.array(labels)
    for lbl in unique_labels:
        mask = label_arr == lbl
        ax.scatter(
            proj[mask, 0], proj[mask, 1], proj[mask, 2],
            s=sizes[mask], color=colors[lbl], label=lbl, alpha=0.8,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Embedding space (PCA 3D)")
    ax.legend()
    return ax
