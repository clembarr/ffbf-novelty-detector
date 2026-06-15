#!/usr/bin/env python3
"""Run from repo root: python notebooks/_build.py"""
import os
import nbformat as nbf

os.makedirs("notebooks", exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

TECH = [
    "GPUs are essential for training deep neural networks.",
    "Rust's borrow checker prevents data races at compile time.",
    "Transformers revolutionized natural language processing.",
    "A Bloom filter tests set membership with probabilistic guarantees.",
    "Moore's law describes the doubling of transistor density roughly every two years.",
    "Kubernetes orchestrates containerized workloads across clusters.",
    "Gradient descent minimizes a loss function by following the negative gradient.",
    "Version control enables collaborative software development through branching.",
    "The Linux kernel is written primarily in C and assembly.",
    "Attention mechanisms allow models to focus on relevant input tokens.",
    "Sparse random projections preserve pairwise distances in high-dimensional space.",
    "TCP provides reliable, ordered delivery of bytes over IP networks.",
    "A hash function maps arbitrary data to fixed-size digest values.",
    "JIT compilation translates bytecode to native machine instructions at runtime.",
    "Convolutional neural networks exploit spatial locality in image data.",
    "Database indexing trades write overhead for faster read queries.",
    "Backpropagation computes gradients via the chain rule of calculus.",
    "WebAssembly enables near-native performance in web browsers.",
    "The CAP theorem states distributed systems cannot guarantee consistency, availability, and partition tolerance simultaneously.",
    "Embeddings map discrete tokens to dense continuous vector representations.",
    "Memory-mapped files allow treating disk regions as in-process virtual memory.",
    "Quantization reduces model size by lowering floating-point precision.",
    "Asymptotic analysis describes algorithm complexity as input size grows.",
    "SIMD instructions process multiple data elements in a single CPU clock cycle.",
    "Reinforcement learning agents maximize cumulative reward through trial and error.",
]

CUISINE = [
    "Al dente pasta requires well-salted boiling water.",
    "Maillard reaction gives browned food its distinctive flavor.",
    "Mise en place means having all ingredients prepped before cooking begins.",
    "Tempering chocolate aligns cocoa butter crystals for a glossy snap.",
    "Emulsification binds fat and water with the help of a lecithin-rich emulsifier.",
    "A roux of equal parts butter and flour thickens sauces and soups.",
    "Deglazing a pan with wine dissolves the caramelized fond into a sauce.",
    "Blanching vegetables preserves color and texture before freezing.",
    "Fermentation by lactic acid bacteria transforms milk into yogurt.",
    "Braising cooks tough cuts of meat low and slow in a small amount of liquid.",
    "Umami is the savory taste associated with glutamate-rich ingredients.",
    "Sous vide cooking seals food in a bag and cooks it in a temperature-controlled water bath.",
    "Reduction concentrates flavor by boiling off liquid from a sauce.",
    "Caramelization occurs when sugars are heated above 160°C.",
    "A beurre blanc sauce is an emulsion of butter, white wine, and shallots.",
    "Fresh herbs added at the end of cooking preserve volatile aromatic compounds.",
    "Salt draws moisture from ingredients through osmosis.",
    "Stock is made by simmering bones and aromatics to extract collagen and flavor.",
    "Kneading dough develops gluten networks that trap carbon dioxide from yeast.",
    "Scoring bread before baking controls oven spring and crust expansion.",
    "Fat-soluble vitamins in vegetables are better absorbed when cooked with oil.",
    "Dry brining meat with salt draws out moisture that is then reabsorbed.",
    "Acid from lemon juice or vinegar brightens flavors and balances richness.",
    "Mirepoix is a base of diced onion, carrot, and celery used in French cooking.",
    "Rest time after roasting allows muscle fibers to relax and reabsorb juices.",
]


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


nb.cells = [
    # ── Title ──────────────────────────────────────────────────────────────────
    md(
        "# 🪰 ffbf — Novelty Detection on a Semantic Stream\n\n"
        "This notebook demonstrates the **Fruit Fly Bloom Filter** on real sentence embeddings.\n"
        "We feed it a stream of tech sentences, watch novelty collapse as patterns become familiar,\n"
        "then introduce a cuisine sentence to trigger a spike — and finally let the filter drift and forget.\n\n"
        "> **Three acts:** learn tech → detect the intruder → drift to cuisine with passive forgetting."
    ),

    # ── § 1 Setup ──────────────────────────────────────────────────────────────
    md("## § 1 — Setup"),
    code("# %pip install sentence-transformers matplotlib  # uncomment if needed"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib\n"
        "from sentence_transformers import SentenceTransformer\n"
        "from ffbf import FFBF, FFBFConfig, DecayMode, TickShape\n"
        "from ffbf.viz import (\n"
        "    plot_novelty_scores,\n"
        "    plot_weight_state,\n"
        "    plot_weight_evolution,\n"
        "    plot_embedding_3d,\n"
        ")\n\n"
        'matplotlib.rcParams["figure.dpi"] = 120\n\n'
        'model = SentenceTransformer("all-MiniLM-L6-v2")\n\n'
        "cfg = FFBFConfig.default_for(input_dim=384, expected_n=50)\n"
        "cfg.delta = 0.4\n"
        "cfg.decay_mode = DecayMode.EdgeAndFront\n"
        "cfg.tick_shape = TickShape.Exp\n"
        "cfg.tick_rate = 0.02\n"
        "cfg.seed = 42\n\n"
        "f = FFBF(cfg)\n"
        "m = len(f.weights())\n"
        'print(f"Filter ready — {m} KC→MBON synapses, window_size={cfg.window_size}")'
    ),

    # ── § 2 Corpus ─────────────────────────────────────────────────────────────
    md("## § 2 — Corpus\n\n25 tech sentences and 25 cuisine sentences, hardcoded."),
    code(
        f"TECH = {repr(TECH)}\n\n"
        f"CUISINE = {repr(CUISINE)}\n\n"
        "tech_emb = model.encode(TECH, convert_to_numpy=True).astype(np.float32)\n"
        "cuisine_emb = model.encode(CUISINE, convert_to_numpy=True).astype(np.float32)\n"
        'print(f"Tech: {tech_emb.shape}, Cuisine: {cuisine_emb.shape}")'
    ),

    # ── § 3 Acte I ─────────────────────────────────────────────────────────────
    md(
        "## § 3 — Act I: Learning Tech\n\n"
        "We feed 25 tech sentences one by one, measuring novelty *before* each `add()`. "
        "Watch the score collapse from ~1.0 toward ~0.1 as patterns become familiar. "
        "The weight state plots show how active synapses get depressed."
    ),
    code(
        "weights_fresh = f.weights().copy()\n"
        "scores_I: list[float] = []\n"
        "for vec in tech_emb:\n"
        "    scores_I.append(f.novelty(vec))\n"
        "    f.add(vec)\n"
        "weights_after_I = f.weights().copy()\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "plot_novelty_scores(scores_I, ax=axes[0])\n"
        'axes[0].set_title("Novelty — Act I (Tech)")\n'
        'plot_weight_state(weights_fresh, title="Fresh filter", ax=axes[1])\n'
        'plot_weight_state(weights_after_I, title="After Act I", ax=axes[2])\n'
        'plt.suptitle("Act I: the filter learns tech", y=1.02)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),

    # ── § 4 Acte II ────────────────────────────────────────────────────────────
    md(
        "## § 4 — Act II: The Intruder\n\n"
        "A single cuisine sentence arrives. The filter has never seen anything like it — novelty spikes."
    ),
    code(
        'spike_sentence = "Les pâtes al dente nécessitent une eau bien salée."\n'
        "vec_spike = model.encode([spike_sentence], convert_to_numpy=True)[0].astype(np.float32)\n"
        "score_spike = f.novelty(vec_spike)\n"
        'print(f"Novelty of intruder: {score_spike:.3f}")\n'
        "#don't add — just observe the score"
    ),

    # ── § 5 Acte III ───────────────────────────────────────────────────────────
    md(
        "## § 5 — Act III: Drift & Forgetting\n\n"
        "25 cuisine sentences, with `tick()` after each. "
        "`tick()` applies passive weight recovery — the filter *forgets* what it hasn't seen recently. "
        "Tech memories fade; cuisine eventually becomes familiar."
    ),
    code(
        "#run a second filter without tick() for comparison\n"
        "f_notick = FFBF(cfg)\n"
        "for vec in tech_emb:\n"
        "    f_notick.add(vec)\n\n"
        "scores_III: list[float] = []\n"
        "scores_III_notick: list[float] = []\n"
        "weight_history: list[np.ndarray] = []\n\n"
        "for vec in cuisine_emb:\n"
        "    scores_III.append(f.novelty(vec))\n"
        "    f.add(vec)\n"
        "    f.tick()\n"
        "    weight_history.append(f.weights().copy())\n\n"
        "    scores_III_notick.append(f_notick.novelty(vec))\n"
        "    f_notick.add(vec)\n\n"
        "weights_after_III = f.weights().copy()\n"
        "weight_history_arr = np.stack(weight_history)  #shape: (25, m)"
    ),
    code(
        "#full novelty timeline — all 3 acts\n"
        "scores_all = scores_I + [score_spike] + scores_III\n"
        "fig, ax = plt.subplots(figsize=(12, 4))\n"
        "plot_novelty_scores(\n"
        "    scores_all,\n"
        '    phases=[(25, "Act II"), (26, "Act III")],\n'
        "    ax=ax,\n"
        ")\n"
        'plt.title("Full novelty timeline — Tech → Intruder → Cuisine")\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    code(
        "#weight snapshots at key moments\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        'plot_weight_state(weights_fresh, title="Fresh filter", ax=axes[0])\n'
        'plot_weight_state(weights_after_I, title="After Act I (tech learned)", ax=axes[1])\n'
        'plot_weight_state(weights_after_III, title="After Act III (cuisine familiar)", ax=axes[2])\n'
        'plt.suptitle("Synaptic memory at key moments", y=1.02)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    code(
        "#weight evolution for 4 sampled KCs during Act III\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "plot_weight_evolution(weight_history_arr, indices=[0, 5, 10, 20], ax=ax)\n"
        'ax.set_title("KC weight evolution during Act III")\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    code(
        "#3D embedding scatter — tech vs cuisine in PCA space\n"
        "all_emb = np.vstack([tech_emb, cuisine_emb])\n"
        'all_labels = ["tech"] * 25 + ["cuisine"] * 25\n'
        "all_novelty = scores_I + scores_III\n\n"
        "fig = plt.figure(figsize=(10, 8))\n"
        'ax_3d = fig.add_subplot(projection="3d")\n'
        "plot_embedding_3d(all_emb, all_labels, all_novelty, ax=ax_3d)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    code(
        "#with vs without tick() — side-by-side\n"
        "scores_all_notick = scores_I + [score_spike] + scores_III_notick\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
        'plot_novelty_scores(scores_all, phases=[(25, "Act II"), (26, "Act III")], ax=axes[0])\n'
        'axes[0].set_title("With tick() — forgetting enabled")\n'
        'plot_novelty_scores(scores_all_notick, phases=[(25, "Act II"), (26, "Act III")], ax=axes[1])\n'
        'axes[1].set_title("Without tick() — no forgetting")\n'
        'plt.suptitle("tick() lets the filter re-detect previously familiar inputs", y=1.02)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),

    # ── § 6 Pour aller plus loin ───────────────────────────────────────────────
    md(
        "## § 6 — Going Further: Parameter Exploration\n\n"
        "Each subsection replays the full scenario with a different hyperparameter configuration."
    ),
    code(
        "def run_scenario(override_cfg: FFBFConfig) -> tuple[list[float], list[float]]:\n"
        '    """Run tech→cuisine scenario, returning (scores_tech, scores_cuisine)."""\n'
        "    flocal = FFBF(override_cfg)\n"
        "    s_tech: list[float] = []\n"
        "    for vec in tech_emb:\n"
        "        s_tech.append(flocal.novelty(vec))\n"
        "        flocal.add(vec)\n"
        "    s_cuisine: list[float] = []\n"
        "    for vec in cuisine_emb:\n"
        "        s_cuisine.append(flocal.novelty(vec))\n"
        "        flocal.add(vec)\n"
        "        flocal.tick()\n"
        "    return s_tech, s_cuisine"
    ),
    md(
        "### 6.1 Impact of `delta` — learning rate\n\n"
        "Higher `delta` depresses active synapses faster → the filter learns sooner."
    ),
    code(
        "deltas = [0.1, 0.3, 0.7]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, d in zip(axes, deltas):\n"
        "    c = FFBFConfig.default_for(input_dim=384, expected_n=50)\n"
        "    c.delta = d\n"
        "    c.decay_mode = DecayMode.EdgeAndFront\n"
        "    c.tick_shape = TickShape.Exp\n"
        "    c.tick_rate = 0.02\n"
        "    c.seed = 42\n"
        "    s_t, s_c = run_scenario(c)\n"
        '    plot_novelty_scores(s_t + s_c, phases=[(25, "Cuisine")], ax=ax)\n'
        '    ax.set_title(f"delta = {d}")\n'
        'plt.suptitle("Impact of delta (learning rate)", y=1.02)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("### 6.2 Impact of `tick_shape` — forgetting curve"),
    code(
        "shapes = [TickShape.Lin, TickShape.Exp, TickShape.Log]\n"
        'shape_names = ["Lin", "Exp", "Log"]\n'
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, shape, name in zip(axes, shapes, shape_names):\n"
        "    c = FFBFConfig.default_for(input_dim=384, expected_n=50)\n"
        "    c.delta = 0.4\n"
        "    c.decay_mode = DecayMode.EdgeAndFront\n"
        "    c.tick_shape = shape\n"
        "    c.tick_rate = 0.02\n"
        "    c.seed = 42\n"
        "    s_t, s_c = run_scenario(c)\n"
        '    plot_novelty_scores(s_t + s_c, phases=[(25, "Cuisine")], ax=ax)\n'
        '    ax.set_title(f"tick_shape = {name}")\n'
        'plt.suptitle("Impact of tick_shape (forgetting curve)", y=1.02)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### 6.3 Impact of `window_size` — adaptive threshold\n\n"
        "`is_novel()` compares the current score against recent history. "
        "A smaller window reacts faster to score changes."
    ),
    code(
        "windows = [10, 50, 100]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, ws in zip(axes, windows):\n"
        "    c = FFBFConfig.default_for(input_dim=384, expected_n=50)\n"
        "    c.delta = 0.4\n"
        "    c.decay_mode = DecayMode.EdgeAndFront\n"
        "    c.tick_shape = TickShape.Exp\n"
        "    c.tick_rate = 0.02\n"
        "    c.window_size = ws\n"
        "    c.seed = 42\n"
        "    flocal = FFBF(c)\n"
        "    novel_flags: list[int] = []\n"
        "    for vec in np.vstack([tech_emb, cuisine_emb]):\n"
        "        novel_flags.append(int(flocal.is_novel(vec, threshold=1.0)))\n"
        "        flocal.add(vec)\n"
        "        flocal.tick()\n"
        '    ax.step(range(len(novel_flags)), novel_flags, where="post")\n'
        "    ax.axvline(x=25, color=\"gray\", linestyle=\"--\", linewidth=1)\n"
        '    ax.text(25.5, 1.05, "Cuisine", fontsize=8, color="gray", va="bottom")\n'
        '    ax.set_xlabel("Step")\n'
        '    ax.set_ylabel("is_novel")\n'
        "    ax.set_ylim(-0.1, 1.3)\n"
        '    ax.set_title(f"window_size = {ws}")\n'
        'plt.suptitle("Impact of window_size (adaptive threshold sensitivity)", y=1.02)\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
]

with open("notebooks/ffbf_semantic_demo.ipynb", "w") as fout:
    nbf.write(nb, fout)
print("✓ notebooks/ffbf_semantic_demo.ipynb created")
