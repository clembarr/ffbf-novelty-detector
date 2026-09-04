from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from mpl_toolkits.mplot3d import Axes3D


PALETTE: dict[str, str] = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "series_3": "#4a3aa7",
}
"""Chart palette: neutral chrome plus three categorical hues, the most an all-pairs scatter keeps separable."""

_SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
"""Single-hue blue ramp, light to dark, used for every magnitude encoding."""


def apply_style() -> None:
    """Apply the notebook chart style to matplotlib rcParams.
    Sets the chart surface, recessive solid hairline grid, muted axis ink and the
    default series color, so every figure in a session reads as one system.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["ink_soft"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "axes.prop_cycle": plt.cycler(
            color=[PALETTE["series_1"], PALETTE["series_2"], PALETTE["series_3"]]
        ),
        "figure.dpi": 130,
    })


def novelty_cmap() -> Colormap:
    """Build the sequential colormap used for novelty and weight magnitudes.
    Returns:
        Colormap: single-hue blue ramp, light (low) to dark (high).
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ffbf_blue", _SEQUENTIAL_BLUE)


def plot_novelty_scores(
    scores: list[float],
    phases: list[tuple[int, str]] | None =None,
    label: str | None =None,
    ax: Axes | None =None
) -> Axes:
    """Line plot of novelty scores over time.
    Parameters:
        scores (list[float]): novelty score at each time step
        phases (list[tuple[int, str]] | None): vertical markers as (step_index, label) pairs
        label (str | None): curve label; adds a legend so several runs can share one axes
        ax (Axes | None): matplotlib axes to draw on; created if None
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(scores, label=label)
    if label is not None:
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Novelty score")
    ax.set_title("Novelty scores over time")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    if phases:
        #alternate a light wash between phase boundaries so consecutive acts read apart
        bounds: list[int] = [idx for idx, _ in phases] + [len(scores) - 1]
        #labels closer than 8% of the stream would overlap, so they drop to a second row
        min_gap: float = 0.08 * len(scores)
        previous_idx: float = -min_gap
        row: int = 0
        for i, (idx, phase_label) in enumerate(phases):
            if i % 2 == 0:
                ax.axvspan(idx, bounds[i + 1], color=PALETTE["grid"], alpha=0.45, linewidth=0)
            ax.axvline(x=idx, color=PALETTE["axis"], linewidth=1)
            row = row + 1 if idx - previous_idx < min_gap else 0
            ax.text(idx + 0.3, 1.02 - 0.07 * row, phase_label, fontsize=8,
                    color=PALETTE["muted"], va="bottom")
            previous_idx = idx
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
    Colors come from the categorical palette slots, in label order; a fourth label and beyond
    falls back to muted gray, since only three slots stay separable on an all-pairs scatter.
    Parameters:
        embeddings (np.ndarray): shape (n, d) float32 embedding matrix
        labels (list[str]): domain label per point, length must equal n, used for color grouping
        novelty_scores (list[float]): novelty score per point in [0, 1], length must equal n, controls marker size
        ax (Axes3D | None): 3D axes; created if None
    Returns:
        Axes3D: the 3D axes used for plotting
    Raises:
        ValueError: if embeddings has fewer than 3 columns, or if labels/novelty_scores length differs from n
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401 — registers "3d" projection
    n: int = len(embeddings)
    if embeddings.shape[1] < 3:
        raise ValueError(f"embeddings must have at least 3 columns, got {embeddings.shape[1]}")
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != embeddings rows {n}")
    if len(novelty_scores) != n:
        raise ValueError(f"novelty_scores length {len(novelty_scores)} != embeddings rows {n}")
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    proj = _pca(embeddings)
    unique_labels = sorted(set(labels))
    slots: list[str] = [PALETTE["series_1"], PALETTE["series_2"], PALETTE["series_3"]]
    colors = {lbl: (slots[i] if i < len(slots) else PALETTE["muted"]) for i, lbl in enumerate(unique_labels)}
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


def plot_novelty_map(
    coords: np.ndarray,
    novelty_scores: list[float] | np.ndarray,
    domains: list[str] | None =None,
    title: str | None =None,
    axis_labels: tuple[str, str] | None =None,
    ax: Axes | None =None,
    colorbar: bool =True,
    legend: bool =True
) -> Axes:
    """Scatter of a 2D projection where color carries the novelty of each point.
    Domains are separated by marker shape, not by hue, so hue stays free for magnitude.
    Parameters:
        coords (np.ndarray): shape (n, 2) projected coordinates, e.g. from UMAP or PCA
        novelty_scores (list[float] | np.ndarray): novelty per point in [0, 1], drives color
        domains (list[str] | None): group label per point, mapped to marker shapes
        title (str | None): axes title
        axis_labels (tuple[str, str] | None): names of the two projection axes, e.g. ("UMAP-1", "UMAP-2");
            worth setting so readers do not mistake a position for the novelty value
        ax (Axes | None): matplotlib axes; created if None
        colorbar (bool): whether to draw the novelty scale beside the axes
        legend (bool): whether to draw the domain-shape legend; turn off on small multiples
    Returns:
        Axes: the axes used for plotting
    Raises:
        ValueError: if coords is not (n, 2), or if novelty_scores/domains length differs from n
    """
    import matplotlib.pyplot as plt
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    n: int = len(coords)
    scores = np.asarray(novelty_scores, dtype=float)
    if len(scores) != n:
        raise ValueError(f"novelty_scores length {len(scores)} != coords rows {n}")
    if domains is not None and len(domains) != n:
        raise ValueError(f"domains length {len(domains)} != coords rows {n}")
    if ax is None:
        _, ax = plt.subplots()
    cmap = novelty_cmap()
    markers: list[str] = ["o", "^", "s", "D", "P"]
    groups: list[str | None] = sorted(set(domains)) if domains is not None else [None]
    domain_arr = np.array(domains) if domains is not None else None
    scatter = None
    for i, group in enumerate(groups):
        mask = np.ones(n, dtype=bool) if domain_arr is None else domain_arr == group
        scatter = ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=scores[mask], cmap=cmap, vmin=0.0, vmax=1.0,
            marker=markers[i % len(markers)], s=90,
            #a surface-colored ring keeps overlapping points readable without a hard border
            linewidths=1.4, edgecolors=PALETTE["surface"],
            label=group,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=8, color=PALETTE["muted"])
        ax.set_ylabel(axis_labels[1], fontsize=8, color=PALETTE["muted"])
    ax.grid(False)
    #equal aspect keeps the projection undistorted; the margin leaves room for the legend
    ax.set_aspect("equal")
    ax.margins(0.14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    if domains is not None and legend:
        handles = ax.legend(loc="best", handletextpad=0.2, borderpad=0.2)
        for handle in handles.legend_handles:
            handle.set_color(PALETTE["muted"])
    if colorbar and scatter is not None:
        bar = ax.figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
        bar.set_label("Novelty", color=PALETTE["ink_soft"], fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(color=PALETTE["muted"], labelcolor=PALETTE["muted"], labelsize=8)
    return ax


def plot_weight_map(
    weights: np.ndarray,
    title: str | None =None,
    ax: Axes | None =None,
    colorbar: bool =True
) -> Axes:
    """Heatmap of the whole synaptic memory, one cell per KC.
    Weights are folded into a near-square grid so thousands of KCs stay readable at a glance.
    Parameters:
        weights (np.ndarray): 1D float32 array of KC weights in [0, w_max]
        title (str | None): axes title; defaults to "Synaptic memory"
        ax (Axes | None): matplotlib axes; created if None
        colorbar (bool): whether to draw the weight scale beside the axes
    Returns:
        Axes: the axes used for plotting
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    m: int = len(weights)
    #a divisor of m near its square root fills the grid exactly, avoiding a ragged last row
    divisors: list[int] = [c for c in range(1, m + 1) if m % c == 0]
    cols: int = min(divisors, key=lambda c: abs(c - np.sqrt(m)))
    rows: int = m // cols
    if cols < 4 or rows < 4:  #a prime-ish m would degenerate into a strip, so pad instead
        cols = int(np.ceil(np.sqrt(m)))
        rows = int(np.ceil(m / cols))
    #pad with NaN so any trailing cell stays blank instead of reading as a depressed synapse
    grid = np.full(rows * cols, np.nan, dtype=float)
    grid[:m] = weights
    image = ax.imshow(
        grid.reshape(rows, cols), cmap=novelty_cmap(), vmin=0.0, vmax=1.0,
        interpolation="nearest", aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title if title is not None else "Synaptic memory")
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        bar.set_label("Weight", color=PALETTE["ink_soft"], fontsize=9)
        bar.outline.set_visible(False)
        bar.ax.tick_params(color=PALETTE["muted"], labelcolor=PALETTE["muted"], labelsize=8)
    return ax
