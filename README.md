# Fruit Fly Bloom Filter (FFBF)

A biologically-inspired novelty detector modelled on the olfactory system of the fruit fly. Unlike a classical Bloom filter (binary yes/no), the **FFBF** produces a **continuous, space and time sensitive, novelty score**.

To fully understand the algorithm, I recommend reading the paper which inspired it first:
[A neural data structure for novelty detection (S.Dasgupta at al., 2018)](https://www.pnas.org/doi/full/10.1073/pnas.1814448115)

The project currently contains a Rust library (`ffbf`), which implements the filter and its persistence, and a Python wrapper for graph analysis and benchmarking.

## How it works

The algorithm mirrors three stages of the fly's olfactory system:

1. **Sparse random projection** - the input vector (Projection Neuron activations) is projected onto a pool of Kenyon Cells (KCs) via a fixed sparse random matrix. Each KC samples a random subset of the input (`projection_sparsity`). Only the top-`k` most activated KCs fire (winner-take-all) a response.
2. **Synaptic weight update** - each firing KC depresses its synapse to the Mushroom Body Output Neuron (MBON) (multiplied by `1 − δ`); non-firing KC synapses recover (`+ε`, capped at `w_max`). The filter tracks `m` KC→MBON synapses in total. Repeated exposure drives active-synapse weights toward zero.
3. **Novelty score** - mean KC→MBON synaptic weight at the `k` active positions, computed *before* the update. A fresh filter yields `1.0`; a fully familiar input approaches `0.0`.

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

`FFBFConfig::default_for(input_dim, expected_n)` provides defaults config. All fields are public and can be overridden before calling `FFBF::new`.

| Field | Default | Range | Effect |
|---|---|---|---|
| `input_dim` | - | > 0 | Input vector length |
| `m` | `30 × n` | > k | Number of KC→MBON synapses tracked (filter memory size) |
| `k` | `m × 0.05` | ]0, m[ | Active KCs per input |
| `projection_sparsity` | `0.12` | ]0, 1] | Fraction of inputs connected per KC |
| `delta` | `0.5` | [0, 1[ | Depression rate of active synapses on `add()` |
| `epsilon` | `0.05` | [0, 1] | Recovery increment of inactive synapses on `add()` |
| `w_max` | `1.0` | ≥ 1.0 | Weight ceiling for inactive synapse recovery |
| `decay_mode` | `EdgeOnly` | - | `EdgeOnly`: activity-driven only; `EdgeAndFront`: also time-driven via `tick()` |
| `tick_shape` | `Lin` | - | `Lin` / `Exp` / `Log` - shape of passive recovery curve |
| `tick_rate` | `0.01` | > 0 | Speed of passive weight recovery per `tick()` call |
| `window_size` | `100` | ≥ 2 | Ring-buffer depth for `is_novel()` adaptive baseline |
| `seed` | `None` | - | Fix RNG seed for deterministic projection matrix |

## Decay modes

**`EdgeOnly`** (default): weights only change on `add()`. Memory is purely activity-driven; `tick()` won't do anything.

**`EdgeAndFront`**: `tick()` additionally moves all weights passively toward `1.0` according to `tick_shape` and `tick_rate`. Familiar inputs eventually become novel again.

```rust
use ffbf::{FFBFConfig, DecayMode, TickShape};

let mut cfg = FFBFConfig::default_for(64, 500);
cfg.decay_mode = DecayMode::EdgeAndFront;
cfg.tick_shape = TickShape::Lin;
cfg.tick_rate = 0.005;
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

The projection matrix is saved explicitly, never re-generated from the seed, so the filter remains coherent across library versions.

