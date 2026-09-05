#!/usr/bin/env python3
"""Generates assets/architecture.svg, the README diagram of one filter step.
Run from the repo root: python assets/_architecture.py
The illustrative activations are drawn from a fixed seed so the file is reproducible.
"""
import random

INK: str = "#0b0b0b"
"""Chart palette, kept identical to ffbf.viz so the diagram and the figures read as one system."""
SOFT: str = "#52514e"
MUTED: str = "#898781"
GRID: str = "#e1e0d9"
AXIS: str = "#c3c2b7"
BLUE: str = "#2a78d6"
ORANGE: str = "#eb6834"
VIOLET: str = "#4a3aa7"
SURFACE: str = "#fcfcfb"

FONT: str = "Helvetica, Arial, sans-serif"
"""Font stack, restricted to faces present everywhere the SVG may be rendered."""

WIDTH: int = 980
HEIGHT: int = 520

rng: random.Random = random.Random(7)
out: list[str] = []


def emit(markup: str) -> None:
    """Append one SVG element to the document being built.
    Parameters:
        markup (str): the element, already serialised
    """
    out.append(markup)


def text(
    x: float,
    y: float,
    lines: list[str],
    size: float =11,
    fill: str =SOFT,
    weight: str ="normal",
    anchor: str ="start",
    line_height: float =15,
) -> None:
    """Emit a text block, one tspan per line so wrapping stays under our control.
    Parameters:
        x (float): left edge, or the anchor point when `anchor` is not "start"
        y (float): baseline of the first line
        lines (list[str]): the lines, in order
        size (float): font size in user units
        fill (str): text color
        weight (str): CSS font weight
        anchor (str): SVG text-anchor
        line_height (float): baseline-to-baseline distance
    """
    spans: str = "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else line_height}">{line}</tspan>'
        for i, line in enumerate(lines)
    )
    emit(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
         f'font-weight="{weight}" text-anchor="{anchor}">{spans}</text>')


emit(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
     f'width="{WIDTH}" height="{HEIGHT}" role="img" '
     f'aria-label="How the FFBF turns one input into one novelty score">')
emit(f'<rect width="{WIDTH}" height="{HEIGHT}" rx="14" fill="{SURFACE}" stroke="{GRID}"/>')
emit('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
     f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{AXIS}"/></marker></defs>')

text(40, 46, ["One input, three stages, one number"], size=17, fill=INK, weight="600")
text(40, 70, ["The filter never stores a vector: it stores how worn out its detectors are."],
     size=12, fill=MUTED)

#stage 1 — the input vector, drawn as a stack of magnitudes to read as dense
text(40, 116, ["① The input"], size=13, fill=INK, weight="600")
for i in range(14):
    emit(f'<rect x="40" y="{132 + i * 15}" width="46" height="11" rx="2" fill="{BLUE}" '
         f'opacity="{rng.uniform(0.25, 1.0):.2f}"/>')
text(40, 364, ["an embedding —", "384 numbers, all meaningful"], size=11, fill=MUTED)

emit(f'<line x1="98" y1="225" x2="180" y2="225" stroke="{AXIS}" stroke-width="1.4" '
     'marker-end="url(#a)"/>')
text(139, 208, ["a fixed", "random wiring"], size=10, fill=MUTED, anchor="middle",
     line_height=12)

#stage 2 — the detectors, winners drawn larger and in the accent hue
text(196, 116, ["② 1 200 detectors"], size=13, fill=INK, weight="600")
winners: set[int] = set(rng.sample(range(140), 14))
for i in range(140):
    cx: int = 204 + (i % 14) * 17
    cy: int = 146 + (i // 14) * 17
    if i in winners:
        emit(f'<circle cx="{cx}" cy="{cy}" r="5.6" fill="{ORANGE}"/>')
    else:
        emit(f'<circle cx="{cx}" cy="{cy}" r="4.4" fill="{GRID}"/>')
text(196, 364, ["each watches a random handful of the input;",
                "only the loudest fire — that is its signature"], size=11, fill=MUTED)

emit(f'<line x1="458" y1="225" x2="534" y2="225" stroke="{AXIS}" stroke-width="1.4" '
     'marker-end="url(#a)"/>')
text(496, 208, ["the ones", "that fired"], size=10, fill=MUTED, anchor="middle", line_height=12)

#stage 3 — the memory, one bar per detector, the winners already worn down
text(548, 116, ["③ The memory"], size=13, fill=INK, weight="600")
emit(f'<line x1="558" y1="140" x2="558" y2="312" stroke="{AXIS}" stroke-width="1"/>')
text(550, 145, ["1.0"], size=9, fill=MUTED, anchor="end")
text(550, 315, ["0"], size=9, fill=MUTED, anchor="end")
worn: set[int] = set(rng.sample(range(30), 6))
for i in range(30):
    weight_value: float = rng.uniform(0.08, 0.3) if i in worn else rng.uniform(0.82, 1.0)
    height: float = weight_value * 170
    emit(f'<rect x="{566 + i * 11.6:.1f}" y="{312 - height:.1f}" width="7" height="{height:.1f}" '
         f'rx="1.5" fill="{ORANGE if i in worn else BLUE}" '
         f'opacity="{1.0 if i in worn else 0.55}"/>')
text(548, 364, ["one weight per detector, all starting at 1.0 —",
                "the whole memory, a few kilobytes, and it never grows"], size=11, fill=MUTED)

#bottom row — the reading that comes out, and the two forces that move the memory
cards: list[tuple[int, str, str, list[str]]] = [
    (40, BLUE, "What comes out",
     ["The average weight of the detectors that", "fired, read before anything is written."]),
    (356, ORANGE, "Learning  ·  add()",
     ["Those weights drop. The input — and anything",
      "that means something close — now reads as known."]),
    (672, VIOLET, "Forgetting  ·  tick()",
     ["Every weight drifts back toward 1.0. What stops", "arriving slowly becomes new again."]),
]
for x, hue, title, body in cards:
    emit(f'<rect x="{x}" y="396" width="296" height="104" rx="10" fill="#ffffff" '
         f'stroke="{GRID}"/>')
    emit(f'<rect x="{x}" y="396" width="4" height="104" rx="2" fill="{hue}"/>')
    text(x + 20, 424, [title], size=12.5, fill=INK, weight="600")
    text(x + 20, 446, body, size=11, fill=SOFT)

#the gauge sits inside the first card: the reading the whole pipeline exists to produce
emit(f'<rect x="60" y="478" width="256" height="7" rx="3.5" fill="{GRID}"/>')
emit(f'<circle cx="{60 + 0.78 * 256:.0f}" cy="481.5" r="6" fill="{BLUE}"/>')
text(60, 496, ["familiar"], size=9, fill=MUTED)
text(316, 496, ["never seen anything like it"], size=9, fill=MUTED, anchor="end")
emit("</svg>")

with open("assets/architecture.svg", "w") as fout:
    fout.write("\n".join(out) + "\n")
print("✓ assets/architecture.svg created")
