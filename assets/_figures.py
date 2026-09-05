#!/usr/bin/env python3
"""Generates the README figures from a log-stream scenario.
Run from the repo root: python assets/_figures.py
The stream is deliberately different from the demo notebook's: routine service logs first,
then a burst of security events, so the README and the notebook illustrate the same
mechanisms on two unrelated corpora.
"""
import matplotlib.pyplot as plt
import numpy as np
import umap
from sentence_transformers import SentenceTransformer

from ffbf import FFBF, FFBFConfig, DecayMode, TickShape
from ffbf.viz import (
    PALETTE,
    apply_style,
    novelty_cmap,
    plot_novelty_map,
    plot_weight_map,
    plot_weight_evolution,
)

ROUTINE: list[str] = [
    "GET /api/v1/orders 200 in 34ms",
    "POST /api/v1/checkout 201 in 118ms",
    "health probe on pod api-7f9c passed",
    "cache hit ratio 0.94 over the last minute",
    "connection pool resized to 24 idle connections",
    "scheduled job nightly-invoices finished in 42s",
    "static asset served from the edge cache in 6ms",
    "worker picked up 12 messages from the email queue",
    "database migration 0142 already applied, skipping",
    "session refreshed for user 40213",
    "GET /api/v1/products?page=3 200 in 51ms",
    "TLS certificate for api.example.com renewed automatically",
    "autoscaler kept the replica count at 4",
    "search index committed 320 documents",
    "PUT /api/v1/cart/9931 204 in 22ms",
    "backup snapshot uploaded to cold storage",
    "feature flag checkout-v2 evaluated for 1200 requests",
    "queue depth is 3 messages, well under the alert threshold",
    "read replica lag steady at 40 milliseconds",
    "config reload completed without restarting the process",
    "DELETE /api/v1/sessions/8821 200 in 18ms",
    "image thumbnailer processed 48 uploads",
    "metrics exporter scraped 1400 series",
    "webhook delivered to the partner endpoint on the first attempt",
    "log rotation archived yesterday's access log",
]
"""Ordinary service traffic: the regime the filter learns first."""

SECURITY: list[str] = [
    "failed password for root from 203.0.113.44 port 51022",
    "sudo: three incorrect password attempts for deploy",
    "unusual outbound connection to 198.51.100.7 on port 4444",
    "ssh login accepted for an account with no recent activity",
    "rate limiter dropped 4200 requests from a single address",
    "query contains a suspicious UNION clause",
    "file integrity monitor detected a change in /etc/shadow",
    "container escaped its seccomp profile and was killed",
    "certificate presented by the peer does not match its hostname",
    "requests carrying a directory traversal pattern were rejected",
    "privilege escalation attempt blocked by the kernel policy",
    "credential stuffing pattern detected across 300 accounts",
    "unexpected process spawned by the web server user",
    "port scan observed across the internal subnet",
    "api token used from two countries within a minute",
    "malformed JWT signature rejected by the gateway",
    "brute force lockout triggered for the admin console",
    "outbound DNS queries to a newly registered domain",
    "web shell signature matched in an uploaded file",
    "audit log tampering suspected: sequence gap detected",
    "unauthorized read of the secrets volume",
    "cross-site scripting payload sanitised from a form field",
    "cryptominer signature found in a running container",
    "firewall denied inbound traffic on an unexpected port",
    "session cookie replayed from a different device fingerprint",
]
"""The second regime: what arrives once the stream turns into an incident."""

LEARNED: int = 20
"""Routine lines fed to the filter; the remaining five probe generalisation."""


def config() -> FFBFConfig:
    """Filter configuration for the README run, tuned like the demo notebook's.
    Returns:
        FFBFConfig: a sensitive filter that learns in one pass and forgets at a visible pace
    """
    cfg: FFBFConfig = FFBFConfig.default_for(input_dim=384, expected_n=50)
    cfg.m = 1200
    cfg.k = 20
    cfg.projection_sparsity = 0.01
    cfg.delta = 0.9
    cfg.epsilon = 0.002
    cfg.decay_mode = DecayMode.EdgeAndFront
    cfg.tick_shape = TickShape.Log
    cfg.tick_rate = 0.06
    cfg.window_size = 20
    cfg.seed = 42
    return cfg


def novelty_of(filt: FFBF, embeddings: np.ndarray) -> np.ndarray:
    """Novelty of every embedding under the current filter state, without modifying it.
    Parameters:
        filt (FFBF): filter to probe
        embeddings (np.ndarray): shape (n, 384) matrix of inputs
    Returns:
        np.ndarray: shape (n,) novelty scores
    """
    return np.array([filt.novelty(v) for v in embeddings])


apply_style()
model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
routine_emb: np.ndarray = model.encode(ROUTINE, convert_to_numpy=True).astype(np.float32)
security_emb: np.ndarray = model.encode(SECURITY, convert_to_numpy=True).astype(np.float32)
all_emb: np.ndarray = np.vstack([routine_emb, security_emb])
DOMAINS: list[str] = ["Routine"] * len(ROUTINE) + ["Security"] * len(SECURITY)

stream: np.ndarray = routine_emb[:LEARNED]
holdout: np.ndarray = routine_emb[LEARNED:]

f: FFBF = FFBF(config())
map_fresh: np.ndarray = novelty_of(f, all_emb)
weight_history: list[np.ndarray] = []
for vec in stream:
    f.add(vec)
    weight_history.append(f.weights().copy())
map_after_learning: np.ndarray = novelty_of(f, all_emb)

#the three probes the README quotes: a learned line, an unseen line of the same kind, an intruder
probe_learned: float = float(novelty_of(f, stream).mean())
probe_unseen: float = float(novelty_of(f, holdout).mean())
probe_intruder: float = float(novelty_of(f, security_emb).mean())

#twin filter on the same stream, never told to forget, to size what tick() buys
f_notick: FFBF = FFBF(config())
for vec in stream:
    f_notick.add(vec)

routine_curve: list[float] = []
security_curve: list[float] = []
routine_curve_notick: list[float] = []
map_mid: np.ndarray = np.zeros(len(all_emb))
for step, vec in enumerate(security_emb):
    f.add(vec)
    f.tick()
    f_notick.add(vec)
    routine_curve.append(float(novelty_of(f, stream).mean()))
    security_curve.append(float(novelty_of(f, security_emb).mean()))
    routine_curve_notick.append(float(novelty_of(f_notick, stream).mean()))
    weight_history.append(f.weights().copy())
    if step == 11:
        map_mid = novelty_of(f, all_emb)
map_after_drift: np.ndarray = novelty_of(f, all_emb)
weight_history_arr: np.ndarray = np.stack(weight_history)

#cosine matches how the embeddings were trained; a large n_neighbors keeps the global
#layout readable on a corpus this small
coords: np.ndarray = umap.UMAP(
    n_components=2,
    n_neighbors=25,
    min_dist=0.6,
    spread=1.5,
    metric="cosine",
    random_state=42,
).fit_transform(all_emb)

#figure 1 — the same fifty lines at four moments, position fixed, only the shade moving
snapshots: list[tuple[str, np.ndarray]] = [
    ("① Fresh — every line is new", map_fresh),
    ("② After 20 routine lines", map_after_learning),
    ("③ Mid-incident", map_mid),
    ("④ Routine traffic is new again", map_after_drift),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for i, (ax, (title, snapshot)) in enumerate(zip(axes.flat, snapshots)):
    plot_novelty_map(
        coords, snapshot, domains=DOMAINS, title=title, ax=ax,
        colorbar=False, legend=i == 0,
        axis_labels=("UMAP-1", "UMAP-2") if i >= 2 else None,
    )
mappable = plt.cm.ScalarMappable(cmap=novelty_cmap(), norm=plt.Normalize(0.0, 1.0))
bar = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.02)
bar.set_label("Novelty", color=PALETTE["ink_soft"], fontsize=9)
bar.outline.set_visible(False)
bar.ax.tick_params(color=PALETTE["muted"], labelcolor=PALETTE["muted"], labelsize=8)
fig.savefig("assets/novelty-map-over-time.png", dpi=150, bbox_inches="tight")
plt.close(fig)

#figure 2 — the two regimes trading places as the stream turns
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(routine_curve, color=PALETTE["series_1"], label="Routine traffic (stopped arriving)")
ax.plot(security_curve, color=PALETTE["series_2"], label="Security events (arriving now)")
ax.set_xlabel("Lines since the stream turned")
ax.set_ylabel("Mean novelty of the group")
ax.set_ylim(0, 1.05)
ax.legend(loc="center right")
ax.set_title("What stops arriving becomes new again")
for curve in (routine_curve, security_curve):
    ax.annotate(f"{curve[-1]:.2f}", (len(curve) - 1, curve[-1]), textcoords="offset points",
                xytext=(6, -3), color=PALETTE["ink"], fontsize=9)
fig.tight_layout()
fig.savefig("assets/drift.png", dpi=150, bbox_inches="tight")
plt.close(fig)

#figure 3 — the memory as a whole, then three synapses followed one at a time
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
plot_weight_map(f.weights(), title="The whole memory, 1 200 weights", ax=axes[0])
#the synapses written most often, not the lowest ones: they are the only ones whose teeth
#show both forces instead of a single late drop
depressions: np.ndarray = (np.diff(weight_history_arr, axis=0) < 0).sum(axis=0)
busiest: list[int] = list(np.argsort(depressions)[-3:])
plot_weight_evolution(weight_history_arr, indices=busiest, ax=axes[1])
axes[1].axvline(x=LEARNED - 0.5, color=PALETTE["axis"], linewidth=1)
axes[1].text(LEARNED - 0.1, 1.02, "the stream turns", fontsize=8, color=PALETTE["muted"],
             va="bottom")
axes[1].set_title("Three synapses, written and forgotten")
axes[1].set_xlabel("Line of the stream")
fig.savefig("assets/memory.png", dpi=150, bbox_inches="tight")
plt.close(fig)

gain_tick: float = routine_curve[-1] - routine_curve[0]
gain_notick: float = routine_curve_notick[-1] - routine_curve_notick[0]
print(f"learned line       {probe_learned:.2f}")
print(f"unseen routine     {probe_unseen:.2f}")
print(f"security lines     {probe_intruder:.2f}  (mean, none of them seen)")
print(f"routine  {routine_curve[0]:.2f} -> {routine_curve[-1]:.2f}")
print(f"security {security_curve[0]:.2f} -> {security_curve[-1]:.2f}")
print(f"routine without tick() {routine_curve_notick[0]:.2f} -> {routine_curve_notick[-1]:.2f}"
      f"  ({gain_tick / gain_notick:.0f}x slower)")
