"""CLI visual identity (doc 10; NFR-002 stdlib-only).

Terminal-native brand assets for FIAE, derived from the same design language
as the hero mark:

- a solid-block ``FIAE`` wordmark (each letter a set of filled cells),
- the single-glyph pixel ``F`` mark (5 wide x 6 tall) that survives favicon /
  menu sizes and doubles as the inline logo,
- ANSI-256 gradient rendering of the wordmark across the signature purple ramp
  (``#C084FC`` -> ``#6D28D9``), with graceful plain-text fallback.

Color policy for a terminal brand shell:

- ``NO_COLOR`` set          -> plain text (https://no-color.org)
- ``FORCE_COLOR`` set       -> force color even if the stream is not a TTY
- stream is not a TTY       -> plain text (piped output stays clean)
- otherwise                 -> ANSI 256
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Block glyphs (solid cells only; matches the SVG matrix-F look)
# ---------------------------------------------------------------------------
_BLOCK_BAR = "\u2588"  # full block

_LETTERS: dict[str, tuple[str, ...]] = {
    "F": (
        "#####", "#....", "#####", "#....", "#....", "#....",
    ),
    "I": (
        "#####", "##...", "##...", "##...", "##...", "#####",
    ),
    "A": (
        ".###.", "#...#", "#####", "#...#", "#...#", "#...#",
    ),
    "E": (
        "#####", "#....", "#####", "#....", "#####", "#....",
    ),
}

# Pixel ``F`` mark: 5 wide x 6 tall, filled stem so it reads at one-glyph size.
_PIXEL_F: tuple[str, ...] = (
    "#####", "#....", "#####", "#....", "#....", "#....",
)


def _cell_char() -> str:
    """Return the filled cell char the current stream can actually encode.

    UTF-8-capable streams get the solid block; on encodings that cannot
    represent it we degrade to ``#`` (which still matches the matrix-F look).
    """
    try:
        enc = sys.stdout.encoding or "utf-8"
        "\u2588".encode(enc)
        return "\u2588"
    except (LookupError, UnicodeEncodeError):
        return "#"


def _render_letter(letter: str, cell: Optional[str] = None) -> tuple[str, ...]:
    """Expand a 5x6 glyph pattern into a rendered row tuple."""
    if cell is None:
        cell = _cell_char()
    return [row.replace("#", cell).replace(".", " ") for row in _LETTERS[letter]]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Brand palette — YOUR exact ANSI-256 codes (as chosen)
# ---------------------------------------------------------------------------
_PURPLE = "\x1b[38;5;141m"      # PURPLE
_DARK = "\x1b[38;5;57m"         # DARK_PURPLE
_BRIGHT = "\x1b[38;5;207m"      # BRIGHT_PURPLE
_RESET = "\x1b[0m"


def _ansi256(index: int) -> str:
    return f"\x1b[38;5;{index}m"


# 256-colour purple ramp (light -> deep) bridging your codes 207 -> 141 -> 57
_PURPLE_RAMP: tuple[int, ...] = (207, 183, 177, 141, 99, 75, 57)


def _ramp_index(step: int, total: int) -> int:
    """Map a column step in [0, total] onto the ANSI-256 purple ramp."""
    if total <= 1:
        pos = 0.0
    else:
        pos = step / (total - 1)
    idx = int(pos * (len(_PURPLE_RAMP) - 1))
    return _PURPLE_RAMP[min(idx, len(_PURPLE_RAMP) - 1)]


def _reset() -> str:
    return "\x1b[0m"


def _supports_color(force: Optional[bool] = None) -> bool:
    """Color availability given environment / stream policy."""
    if force is True:
        return True
    if force is False:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:  # pragma: no cover - exotic streams
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def banner(force_color: Optional[bool] = None) -> str:
    """Return the ``FIAE`` wordmark with a per-column purple gradient."""
    glyphs = [_render_letter(ch) for ch in "FIAE"]
    height = len(glyphs[0])
    chars_per_letter = 5
    spacing = 1
    total_cols = chars_per_letter * len(glyphs) + spacing * (len(glyphs) - 1)

    if not _supports_color(force_color):
        lines = []
        for r in range(height):
            line_parts = []
            for gi, g in enumerate(glyphs):
                line_parts.append(g[r])
                if gi < len(glyphs) - 1:
                    line_parts.append(" ")
            lines.append("".join(line_parts))
        return "\n".join(lines)

    lines = []
    for r in range(height):
        out: list[str] = []
        col = 0
        for gi, g in enumerate(glyphs):
            for c in range(len(g[r])):
                ch = g[r][c]
                if ch != " ":
                    out.append(_ansi256(_ramp_index(col, total_cols)) + ch)
                else:
                    out.append(" ")
                col += 1
            if gi < len(glyphs) - 1:
                out.append(" ")
                col += 1
        lines.append("".join(out) + _reset())
    return "\n".join(lines)


def pixel_f(force_color: Optional[bool] = None) -> str:
    """Return the single-glyph pixel ``F`` mark (plain or coloured)."""
    cell = _cell_char()
    rows = [r.replace("#", cell).replace(".", " ") for r in _PIXEL_F]
    if not _supports_color(force_color):
        return "\n".join(rows)
    painted = [
        "".join(_PURPLE + ch if ch == cell else " " for ch in row)
        for row in rows
    ]
    return "\n".join([*painted, _reset()])


def version_line() -> str:
    """One-line brand + version, ideal for ``fiae --version`` / prompt."""
    from . import __version__  # type: ignore[attr-defined]
    return f"fiae {__version__} - Feature Intelligence & Architecture Engine"


# ---------------------------------------------------------------------------
# Figlet banner (lowercase "fiae" + bordered tagline box)
# ---------------------------------------------------------------------------
def _can_unicode() -> bool:
    """True when the stream can encode the banner's box-drawing glyph set."""
    try:
        enc = sys.stdout.encoding or "utf-8"
        "\u2588\u2550\u2554\u2557\u255a\u255d\u2022".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


_FIGLET_LOGO = "\n".join((
    "     \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557",
    "     \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d",
    "     \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2557  ",
    "     \u2588\u2588\u2554\u2550\u2550\u255d  \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u255d  ",
    "     \u2588\u2588\u2551     \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557",
    "     \u255a\u2550\u255d     \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d",
))


def figlet(
    force_color: Optional[bool] = None,
    with_box: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render the lowercase ``fiae`` figlet, optionally with a bordered box.

    On a stream that cannot encode the box-drawing set, falls back to the
    plain block ``FIAE`` wordmark so nothing is ever garbled.
    """
    if not _can_unicode():
        return banner(force_color=force_color)

    from . import __version__  # type: ignore[attr-defined]

    color = _supports_color(force_color)
    if color:
        logo = "\n".join(f"{_PURPLE}{line}{_reset()}" for line in _FIGLET_LOGO.splitlines())
    else:
        logo = _FIGLET_LOGO

    if not with_box:
        return logo

    box_w = 52 if width is None else max(40, width)

    # NOTE: bar must be built outside the f-string expressions — backslash
    # escapes inside f-string expression parts are a SyntaxError on 3.10/3.11.
    bar = "\u2550" * box_w

    def top() -> str:
        return f"{_DARK}\u2554{bar}\u2557{_reset()}" if color else \
            f"\u2554{bar}\u2557"

    def bot() -> str:
        return f"{_DARK}\u255a{bar}\u255d{_reset()}" if color else \
            f"\u255a{bar}\u255d"

    def row(text: str, dim: bool = False) -> str:
        pad = box_w - 2 - len(text)
        if pad < 0:
            text = text[: box_w - 3] + "\u2026"
            pad = 0
        body = f" {text}{' ' * pad} "
        if color:
            inner = f"{_DARK if dim else _PURPLE}{body}{_reset()}"
            return f"{_DARK}\u2551{_reset()}{inner}{_DARK}\u2551{_reset()}"
        return f"\u2551{body}\u2551"

    title = "FEATURE INTELLIGENCE & ARCHITECTURE ENGINE"
    caption = f"fiae {__version__}  \u2022  built for safety, speed & trust"

    return "\n".join([logo, top(), row(title), row(caption, dim=True), bot()])


def splash(
    tagline: bool = True,
    force_color: Optional[bool] = None,
) -> str:
    """Full CLI splash: lowercase ``fiae`` figlet + bordered tagline box."""
    return figlet(force_color=force_color, with_box=tagline)
