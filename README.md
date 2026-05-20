# ffbf — Fruit Fly Bloom Filter

A biologically-inspired novelty detector modelled on the olfactory circuit of *Drosophila melanogaster*. Unlike a classical Bloom filter (binary yes/no), `ffbf` produces a **continuous novelty score**: high for genuinely new inputs, low for familiar ones, with tunable memory decay over time.

## How it works

The algorithm mirrors three stages of the fly's olfactory system:

1. **Sparse random projection** — each input vector is projected onto `m` Kenyon Cells (KCs) via a fixed sparse random matrix; only the top-`k` most activated KCs fire (winner-take-all).
2. **Synaptic weight update** — active KC synapses are depressed (multiplied by `1 − δ`); inactive ones recover (`+ε`, capped at `w_max`). Repeated exposure drives active-synapse weights toward zero.
3. **Novelty score** — mean weight at active KC positions, computed *before* the update. A fresh filter yields `1.0`; a fully familiar input approaches `0.0`.

An optional `tick()` call applies passive time-based recovery, so old memories fade and previously familiar inputs gradually become novel again.

## Quick start

```rust
use ffbf::{FFBF, FFBFConfig};

// Config tuned for ~1 000 expected elements of dimension 128
let cfg = FFBFConfig::default_for(128, 1_000);
let mut filter = FFBF::new(cfg)?;

let input: Vec<f32> = vec![0.1, 0.4, /* … 128 values … */];

// Score without modifying state
let score = filter.novelty(&input);     // → ~1.0 on a fresh filter

// Record the input and update weights
filter.add(&input);

// Adaptive threshold against recent history
if filter.is_novel(&input, 1.0) {
    println!("novel");
}
```

## Configuration

`FFBFConfig::default_for(input_dim, expected_n)` provides sensible defaults. All fields are public and can be overridden before calling `FFBF::new`.

| Field | Default | Range | Effect |
|---|---|---|---|
| `input_dim` | — | > 0 | Input vector length |
| `m` | `30 × n` | > k | Filter size (KC count) |
| `k` | `m × 0.05` | < m | Active KCs per input |
| `projection_sparsity` | `0.12` | (0, 1] | Fraction of inputs connected per KC |
| `delta` | `0.5` | [0, 1) | Depression rate of active synapses on `add()` |
| `epsilon` | `0.05` | [0, 1] | Recovery increment of inactive synapses on `add()` |
| `w_max` | `2.0` | ≥ 1.0 | Weight ceiling; headroom for reminiscence overshoot |
| `decay_mode` | `EdgeOnly` | — | `EdgeOnly`: activity-driven only; `EdgeAndFront`: also time-driven via `tick()` |
| `tick_shape` | `Exp` | — | `Lin` / `Exp` / `Log` — shape of passive recovery curve |
| `tick_rate` | `0.01` | > 0 | Speed of passive weight recovery per `tick()` call |
| `reminiscence_factor` | `0.0` | [0, 1] | Overshoot above 1.0 for heavily-depressed synapses during `tick()` |
| `window_size` | `100` | ≥ 2 | Ring-buffer depth for `is_novel()` adaptive baseline |
| `seed` | `None` | — | Fix RNG seed for deterministic projection matrix |

## Decay modes

**`EdgeOnly`** (default): weights only change on `add()`. Memory is purely activity-driven; calling `tick()` is a no-op.

**`EdgeAndFront`**: `tick()` additionally moves all weights passively toward `1.0` according to `tick_shape` and `tick_rate`. Familiar inputs eventually become novel again. The `reminiscence_factor` optionally creates a transient overshoot above `1.0` for strongly depressed synapses.

```rust
use ffbf::{FFBFConfig, DecayMode, TickShape};

let mut cfg = FFBFConfig::default_for(64, 500);
cfg.decay_mode = DecayMode::EdgeAndFront;
cfg.tick_shape = TickShape::Log;   // slow approach, long reminiscence tail
cfg.tick_rate = 0.005;
cfg.reminiscence_factor = 0.3;
```

## Persistence

The full filter state (weights, projection matrix, novelty window) serialises to JSON.

```rust
use ffbf::{save, load, to_json, from_json};

// File
save(&filter, "filter.json")?;
let filter = load("filter.json")?;

// In-memory string
let json = to_json(&filter)?;
let filter = from_json(&json)?;
```

The projection matrix is saved explicitly — never re-generated from the seed — so the filter remains coherent across library versions.

## Status

`v0.1` — core library, pure Rust, no `unsafe`. Python bindings (PyO3 / maturin) are planned for `v0.2`.
