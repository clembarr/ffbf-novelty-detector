import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from ffbf.viz import plot_novelty_scores, plot_weight_state, plot_weight_evolution, plot_embedding_3d


def test_plot_novelty_scores_creates_axes():
    import matplotlib.pyplot as plt
    ax = plot_novelty_scores([0.9, 0.7, 0.5, 0.4, 0.4])
    assert ax is not None
    plt.close("all")


def test_plot_novelty_scores_accepts_ax():
    import matplotlib.pyplot as plt
    _, ax_in = plt.subplots()
    ax_out = plot_novelty_scores([0.5, 0.4], ax=ax_in)
    assert ax_out is ax_in
    plt.close("all")


def test_plot_weight_state_creates_axes():
    import matplotlib.pyplot as plt
    weights = np.random.rand(200).astype(np.float32)
    ax = plot_weight_state(weights)
    assert ax is not None
    plt.close("all")


def test_plot_weight_state_accepts_title():
    import matplotlib.pyplot as plt
    weights = np.ones(50, dtype=np.float32) * 0.5
    ax = plot_weight_state(weights, title="After Act I")
    assert ax.get_title() == "After Act I"
    plt.close("all")


def test_plot_weight_state_accepts_ax():
    import matplotlib.pyplot as plt
    _, ax_in = plt.subplots()
    weights = np.random.rand(100).astype(np.float32)
    ax_out = plot_weight_state(weights, ax=ax_in)
    assert ax_out is ax_in
    plt.close("all")


def test_plot_weight_evolution_creates_axes():
    import matplotlib.pyplot as plt
    history = np.random.rand(20, 50).astype(np.float32)
    ax = plot_weight_evolution(history, indices=[0, 1, 2])
    assert ax is not None
    plt.close("all")


def test_plot_weight_evolution_all_indices():
    import matplotlib.pyplot as plt
    history = np.random.rand(5, 10).astype(np.float32)
    ax = plot_weight_evolution(history)
    assert ax is not None
    plt.close("all")


def test_plot_novelty_scores_with_phases():
    import matplotlib.pyplot as plt
    ax = plot_novelty_scores(
        [0.9, 0.8, 0.5, 0.4, 0.9, 0.7, 0.4],
        phases=[(3, "Phase II"), (4, "Phase III")],
    )
    #axvlines are Line2D objects; data line + 2 phase lines = 3 lines minimum
    assert len(ax.get_lines()) >= 3
    plt.close("all")


def test_plot_novelty_scores_phases_none_unchanged():
    import matplotlib.pyplot as plt
    ax = plot_novelty_scores([0.9, 0.5, 0.2], phases=None)
    assert len(ax.get_lines()) == 1
    plt.close("all")


def test_plot_embedding_3d_creates_axes():
    import matplotlib.pyplot as plt
    embeddings = np.random.rand(20, 8).astype(np.float32)
    labels = ["tech"] * 10 + ["cuisine"] * 10
    novelty = [0.5] * 20
    ax = plot_embedding_3d(embeddings, labels, novelty)
    assert ax is not None
    plt.close("all")


def test_plot_embedding_3d_accepts_ax():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax_in = fig.add_subplot(projection="3d")
    embeddings = np.random.rand(10, 6).astype(np.float32)
    labels = ["a"] * 5 + ["b"] * 5
    ax_out = plot_embedding_3d(embeddings, labels, [0.5] * 10, ax=ax_in)
    assert ax_out is ax_in
    plt.close("all")


def test_plot_embedding_3d_raises_on_too_few_columns():
    with pytest.raises(ValueError, match="at least 3 columns"):
        plot_embedding_3d(
            np.ones((5, 2), dtype=np.float32),
            ["a"] * 5,
            [0.5] * 5,
        )


def test_plot_embedding_3d_raises_on_labels_length_mismatch():
    with pytest.raises(ValueError, match="labels length"):
        plot_embedding_3d(
            np.ones((5, 4), dtype=np.float32),
            ["a"] * 3,
            [0.5] * 5,
        )


def test_plot_embedding_3d_raises_on_novelty_length_mismatch():
    with pytest.raises(ValueError, match="novelty_scores length"):
        plot_embedding_3d(
            np.ones((5, 4), dtype=np.float32),
            ["a"] * 5,
            [0.5] * 3,
        )


def test_plot_novelty_scores_label_enables_legend():
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    plot_novelty_scores([0.9, 0.7], label="Lin", ax=ax)
    plot_novelty_scores([0.8, 0.6], label="Exp", ax=ax)
    assert [line.get_label() for line in ax.get_lines()] == ["Lin", "Exp"]
    plt.close("all")


def test_apply_style_sets_surface_and_grid():
    import matplotlib.pyplot as plt
    from ffbf.viz import apply_style, PALETTE
    apply_style()
    assert plt.rcParams["figure.facecolor"] == PALETTE["surface"]
    assert plt.rcParams["grid.color"] == PALETTE["grid"]
    assert plt.rcParams["grid.linestyle"] == "-"


def test_novelty_cmap_runs_light_to_dark():
    from ffbf.viz import novelty_cmap
    cmap = novelty_cmap()
    low, high = cmap(0.0), cmap(1.0)
    #luminance must decrease as novelty grows (sequential, one hue, light -> dark)
    assert sum(low[:3]) > sum(high[:3])


def test_plot_novelty_map_marks_every_point():
    import matplotlib.pyplot as plt
    from ffbf.viz import plot_novelty_map
    coords = np.random.rand(12, 2)
    ax = plot_novelty_map(coords, [0.5] * 12, domains=["tech"] * 7 + ["cuisine"] * 5)
    #one PathCollection per domain, covering all points
    assert sum(len(c.get_offsets()) for c in ax.collections) == 12
    plt.close("all")


def test_plot_novelty_map_raises_on_length_mismatch():
    from ffbf.viz import plot_novelty_map
    with pytest.raises(ValueError):
        plot_novelty_map(np.random.rand(5, 2), [0.5] * 3)


def test_plot_novelty_map_raises_on_non_2d_coords():
    from ffbf.viz import plot_novelty_map
    with pytest.raises(ValueError):
        plot_novelty_map(np.random.rand(5, 3), [0.5] * 5)


def test_plot_weight_map_pads_to_full_grid():
    import matplotlib.pyplot as plt
    from ffbf.viz import plot_weight_map
    ax = plot_weight_map(np.linspace(0, 1, 30, dtype=np.float32))
    grid = ax.images[0].get_array()
    assert grid.size >= 30 and grid.shape[0] * grid.shape[1] == grid.size
    plt.close("all")


def test_plot_novelty_scores_shades_alternate_phase_segments():
    import matplotlib.pyplot as plt
    ax = plot_novelty_scores(
        [0.9, 0.8, 0.5, 0.4, 0.9, 0.7, 0.4],
        phases=[(3, "Act II"), (4, "Act III")],
    )
    #one shaded span for the first segment, none for the second
    assert len(ax.patches) == 1
    plt.close("all")


def test_plot_novelty_scores_staggers_close_phase_labels():
    import matplotlib.pyplot as plt
    ax = plot_novelty_scores(
        list(np.linspace(1.0, 0.2, 40)),
        phases=[(20, "Intruder"), (21, "Act III")],
    )
    heights = [text.get_position()[1] for text in ax.texts]
    #labels one step apart must not sit at the same height
    assert heights[0] != heights[1]
    plt.close("all")


def test_plot_novelty_map_legend_can_be_suppressed():
    import matplotlib.pyplot as plt
    from ffbf.viz import plot_novelty_map
    ax = plot_novelty_map(
        np.random.rand(6, 2), [0.5] * 6, domains=["a"] * 3 + ["b"] * 3, legend=False
    )
    assert ax.get_legend() is None
    plt.close("all")


def test_plot_novelty_map_labels_projection_axes():
    import matplotlib.pyplot as plt
    from ffbf.viz import plot_novelty_map
    ax = plot_novelty_map(
        np.random.rand(6, 2), [0.5] * 6, axis_labels=("UMAP-1", "UMAP-2")
    )
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("UMAP-1", "UMAP-2")
    plt.close("all")
