#!/usr/bin/env python3
"""Strips assets/ffbf-bench.html down to the fragment the Claude Artifact host expects.
Run from the repo root: python assets/_artifact.py [out]
The bench is committed as a complete standalone document, because GitHub Pages serves it as
one. The Artifact host instead wraps whatever it is given in its own <!doctype>/<head>/<body>
skeleton, so publishing the standalone file verbatim nests two documents and breaks the page.
This script removes exactly the skeleton and leaves the rest byte for byte, so the published
artifact and the page in the repo never drift apart.
"""
import os
import re
import sys

SOURCE: str = "assets/ffbf-bench.html"
"""The standalone document, the single source of truth for both outputs."""

DEFAULT_OUT: str = "build/ffbf-bench.fragment.html"
"""Derived output, under the already gitignored build/ so it never reaches a commit."""

SKELETON: tuple[str, ...] = (
    r"<!doctype[^>]*>",
    r"</?html(?:\s[^>]*)?>",
    r"</?head(?:\s[^>]*)?>",
    r"</?body(?:\s[^>]*)?>",
    r"<meta\s+charset=[^>]*>",
    r'<meta\s+name="viewport"[^>]*>',
)
"""Tags the host supplies itself. <title>, <link> and <style> are deliberately absent: the host
keeps them, and the title is what names the artifact in the tab and the gallery."""

FORBIDDEN: tuple[str, ...] = (
    r"<!doctype\b",
    r"</?html\b",
    r"</?head(?:\s|>|/)",
    r"</?body(?:\s|>|/)",
)
"""Anything left of the skeleton after the strip means the source drifted from what this script
knows how to parse — better to fail than to publish a broken page. Matched as tags rather than
substrings, since the page is full of legitimate <header> elements."""


def strip(document: str) -> str:
    """Remove the document skeleton and collapse the blank lines it leaves behind.
    Parameters:
        document (str): the full standalone HTML file
    Returns:
        str: the fragment, with everything else untouched
    Raises:
        SystemExit: when a skeleton tag survives the strip, i.e. the source changed shape.
    """
    fragment: str = document
    for pattern in SKELETON:
        fragment = re.sub(pattern, "", fragment, flags=re.IGNORECASE)
    #the removals leave the lines that held them empty; runs of two or more blank lines are
    #collapsed to one, so the single blank lines the author uses to space the source survive
    fragment = re.sub(r"\n{3,}", "\n\n", fragment).strip() + "\n"

    for pattern in FORBIDDEN:
        left: re.Match[str] | None = re.search(pattern, fragment, flags=re.IGNORECASE)
        if left is not None:
            sys.exit(f"{SOURCE}: '{left.group()}' survived the strip — update SKELETON in {__file__}")
    return fragment


out: str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT

with open(SOURCE, encoding="utf-8") as fp:
    source: str = fp.read()

fragment: str = strip(source)

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w", encoding="utf-8") as fp:
    fp.write(fragment)

print(f"{out} — {len(fragment)} chars, {len(source) - len(fragment)} stripped")
