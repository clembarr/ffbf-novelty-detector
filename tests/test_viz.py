import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from ffbf.viz import plot_novelty_scores, plot_weight_state, plot_weight_evolution


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
