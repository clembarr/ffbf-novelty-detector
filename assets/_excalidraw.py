#!/usr/bin/env python3
"""Generates assets/ffbf-structure.excalidraw, one add() step worked through on an example input.
Run from the repo root: python assets/_excalidraw.py
The result is a plain Excalidraw scene: open or import it on https://excalidraw.com to edit
it by hand. Every stage carries a glyph rather than a description, so the schema reads at a
stays reproducible.
"""
import json
import random
from typing import Any

INK: str = "#0b0b0b"
"""Palette, kept identical to ffbf.viz and assets/architecture.svg so every asset reads as one system."""
SOFT: str = "#52514e"
MUTED: str = "#898781"
GRID: str = "#e1e0d9"
AXIS: str = "#c3c2b7"
BLUE: str = "#2a78d6"
ORANGE: str = "#eb6834"
VIOLET: str = "#4a3aa7"
SURFACE: str = "#fcfcfb"
TRANSPARENT: str = "transparent"

HAND: int = 1
"""Excalidraw font ids: 1 = Excalifont (hand drawn), 2 = Nunito, 3 = Cascadia (monospace)."""

BOX_W: int = 150
BOX_H: int = 110
BOX_Y: int = 200
STAGE_X: list[int] = [60, 250, 440, 630, 820, 1010]
"""Left edge of each pipeline stage, one step every 190 units."""

rng: random.Random = random.Random(7)
elements: list[dict[str, Any]] = []
_counter: int = 0


def _base(kind: str, x: float, y: float, w: float, h: float, group: str | None) -> dict[str, Any]:
    """Build the field set every Excalidraw element must carry.
    Parameters:
        kind (str): element type ('rectangle', 'diamond', 'ellipse', 'text', 'arrow', 'line')
        x, y (float): top-left corner in scene coordinates
        w, h (float): bounding box size
        group (str | None): group id, so a stage and its glyph move together in the editor
    Returns:
        dict[str, Any]: the common element payload, to be completed by the caller
    """
    global _counter
    _counter += 1
    return {
        "id": f"el{_counter:04d}",
        "type": kind,
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0,
        "strokeColor": INK,
        "backgroundColor": TRANSPARENT,
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [group] if group else [],
        "frameId": None,
        "roundness": None,
        "seed": rng.randint(1, 2 ** 31),
        "version": 1,
        "versionNonce": rng.randint(1, 2 ** 31),
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }


def shape(kind: str, x: float, y: float, w: float, h: float,
    stroke: str =INK, bg: str =TRANSPARENT, width: int =1, style: str ="solid",
    radius: bool =False, opacity: int =100, group: str | None =None
) -> dict[str, Any]:
    """Emit a rectangle, diamond or ellipse.
    Parameters:
        kind (str): 'rectangle', 'diamond' or 'ellipse'
        stroke (str): border colour
        bg (str): fill colour, 'transparent' for an outline only
        radius (bool): rounded corners, rectangles only
    Returns:
        dict[str, Any]: the element, already appended to the scene
    """
    e: dict[str, Any] = _base(kind, x, y, w, h, group)
    e.update(strokeColor=stroke, backgroundColor=bg, strokeWidth=width, strokeStyle=style,
             opacity=opacity, roundness={"type": 3} if radius else None)
    elements.append(e)
    return e


def text(x: float, y: float, content: str,
    size: int =18, color: str =INK, group: str | None =None
) -> dict[str, Any]:
    """Emit a hand-written label, sizing its box from a character width estimate.
    Excalidraw stores the box rather than measuring on load, so an estimate keeps neighbouring
    elements clear until the user edits the text.
    Returns:
        dict[str, Any]: the element, already appended to the scene
    """
    e: dict[str, Any] = _base("text", x, y, len(content) * size * 0.52, size * 1.25, group)
    e.update(strokeColor=color, text=content, originalText=content,
             fontSize=size, fontFamily=HAND, textAlign="left",
             verticalAlign="top", containerId=None, lineHeight=1.25)
    elements.append(e)
    return e


def path(points: list[tuple[float, float]], kind: str ="arrow",
    stroke: str =INK, width: int =1, group: str | None =None
) -> dict[str, Any]:
    """Emit an arrow or a line through absolute scene points.
    Parameters:
        kind (str): 'arrow' for a headed connector, 'line' for a bare stroke
    Returns:
        dict[str, Any]: the element, already appended to the scene
    """
    ox, oy = points[0]
    rel: list[list[float]] = [[px - ox, py - oy] for px, py in points]
    e: dict[str, Any] = _base(kind, ox, oy,
                              max(p[0] for p in rel) - min(p[0] for p in rel),
                              max(p[1] for p in rel) - min(p[1] for p in rel), group)
    e.update(strokeColor=stroke, strokeWidth=width, points=rel, lastCommittedPoint=None,
             startBinding=None, endBinding=None, startArrowhead=None,
             endArrowhead="arrow" if kind == "arrow" else None,
             roundness={"type": 2})
    elements.append(e)
    return e



# ---------------------------------------------------------------- the worked example
INPUT: list[float] = [0.9, 0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.3]
"""One concrete input vector, small enough that every sum can be checked by eye."""

WIRING: list[list[int]] = [[0, 1, 3], [0, 2, 4], [1, 5, 7], [2, 4, 6], [3, 5, 6], [1, 4, 7]]
"""connections[kc]: the fixed input lines each KC reads, three out of eight here."""

WEIGHTS: dict[int, float] = {1: 0.80, 3: 0.42}
"""Current synaptic weight of the two winners, the only ones the score ever reads."""

DELTA: float = 0.5
K: int = 2

acts: list[float] = [sum(INPUT[i] for i in conns) for conns in WIRING]
winners: list[int] = sorted(sorted(range(len(acts)), key=lambda kc: -acts[kc])[:K])
score: float = sum(WEIGHTS[kc] for kc in winners) / K

IN_X, IN_W, IN_H, IN_PITCH, IN_TOP = 80, 86, 38, 46, 150
KC_X, KC_W, KC_H, KC_PITCH, KC_TOP = 430, 110, 40, 60, 160
ACT_X, ACT_MAX = 560, 140

in_mid: list[float] = [IN_TOP + i * IN_PITCH + IN_H / 2 for i in range(len(INPUT))]
kc_mid: list[float] = [KC_TOP + j * KC_PITCH + KC_H / 2 for j in range(len(WIRING))]

text(80, 56, "FFBF · add(input)", 22, MUTED)
for cx, label in [(IN_X, "input"), (250, "mapping"), (KC_X, "kc"),
                  (ACT_X, "activation"), (752, "top-k"), (810, "weight"), (930, "score")]:
    text(cx, 108, label, 16, MUTED)

# ---------------------------------------------------------------- input vector
g: str = "input"
for i, v in enumerate(INPUT):
    y: float = IN_TOP + i * IN_PITCH
    text(44, y + 10, f"x{i}", 14, MUTED, g)
    shape("rectangle", IN_X, y, IN_W, IN_H, AXIS, BLUE, 1, "solid", True, int(12 + 80 * v), g)
    text(IN_X + 26, y + 9, f"{v:.1f}", 17, INK, g)

# ---------------------------------------------------------------- mapping: which lines each KC reads
g = "mapping"
for kc, conns in enumerate(WIRING):
    hot: bool = kc in winners
    for i in conns:
        #the path a value actually takes: orange where it ends up feeding the score
        path([(IN_X + IN_W + 4, in_mid[i]),
              ((IN_X + IN_W + KC_X) / 2, (in_mid[i] + kc_mid[kc]) / 2),
              (KC_X - 4, kc_mid[kc])],
             "line", ORANGE if hot else MUTED, 2 if hot else 1, g)
        elements[-1]["opacity"] = 100 if hot else 45

# ---------------------------------------------------------------- kcs and their activation
for kc, conns in enumerate(WIRING):
    g = f"kc{kc}"
    hot = kc in winners
    y = KC_TOP + kc * KC_PITCH
    shape("rectangle", KC_X, y, KC_W, KC_H, ORANGE if hot else AXIS, SURFACE,
          2 if hot else 1, "solid", True, 100, g)
    text(KC_X + 16, y + 10, f"kc{kc}", 17, INK, g)
    width: float = acts[kc] / max(acts) * ACT_MAX
    shape("rectangle", ACT_X, kc_mid[kc] - 8, width, 16, TRANSPARENT,
          ORANGE if hot else BLUE, 1, "solid", False, 100 if hot else 55, g)
    text(ACT_X + ACT_MAX + 12, kc_mid[kc] - 9, f"{acts[kc]:.1f}", 17, INK if hot else SOFT, g)
    if not hot:
        continue
    shape("ellipse", 756, kc_mid[kc] - 6, 12, 12, TRANSPARENT, ORANGE, 1, "solid", False, 100, g)
    shape("rectangle", 810, kc_mid[kc] - 16, 76, 32, AXIS, SURFACE, 1, "solid", True, 100, g)
    text(824, kc_mid[kc] - 8, f"{WEIGHTS[kc]:.2f}", 17, INK, g)
    #the write side of the same step: the winners, and only they, get carved
    path([(848, kc_mid[kc] + 20), (848, kc_mid[kc] + 44)], "arrow", ORANGE, 1, g)
    shape("rectangle", 810, kc_mid[kc] + 46, 76, 32, ORANGE, SURFACE, 1, "dashed", True, 100, g)
    text(824, kc_mid[kc] + 54, f"{WEIGHTS[kc] * (1 - DELTA):.2f}", 17, ORANGE, g)

text(762, kc_mid[winners[0]] + 26, "learn", 16, ORANGE)

# ---------------------------------------------------------------- score
g = "score"
for kc in winners:
    path([(890, kc_mid[kc]), (914, (kc_mid[kc] + 293) / 2), (926, 293)], "arrow", SOFT, 2, g)
shape("rectangle", 930, 250, 176, 86, VIOLET, SURFACE, 2, "solid", True, 100, g)
text(946, 266, " + ".join(f"{WEIGHTS[kc]:.2f}" for kc in winners) + f"  /  {K}", 15, SOFT, g)
text(946, 288, f"{score:.2f}", 32, VIOLET, g)

scene: dict[str, Any] = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://github.com/clembarr/ffbf-novelty-detector",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

with open("assets/ffbf-structure.excalidraw", "w", encoding="utf-8") as fp:
    json.dump(scene, fp, indent=1)

print(f"assets/ffbf-structure.excalidraw — {len(elements)} elements")
