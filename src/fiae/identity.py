"""FIAE visual identity — terminal-native expression of the FIAE brand system.

The GUI/README brand (``logo.svg``) and this terminal identity share one
brand system — palette, mark idea, wordmark geometry — but each surface
uses its native rendering technology:

    VECTOR / GUI                     TERMINAL / CLI
    smooth geometry                  block geometry
    SVG paths + gradients            Unicode blocks + ANSI color
    (logo.svg)                       (this module)

Core requirements (doc 10, NFR-002):
    - zero third-party dependencies (stdlib only)
    - NO_COLOR / dumb-term / non-TTY graceful degradation
    - deterministic output; safe to redirect (nothing prints here)

Brand palette: #C084FC (light) -> #9333EA (mid) -> #6D28D9 (deep),
rendered on 256-color terminals via the classic ANSI-256 ramp
(207 -> 141 -> 57) for maximum compatibility.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

PRODUCT_NAME = "FIAE"
PRODUCT_LONG_NAME = "FEATURE INTELLIGENCE & ARCHITECTURE ENGINE"
PRODUCT_TAGLINE = "Safety-first feature engineering"

# ---------------------------------------------------------------------------
# Brand palette (truecolor anchors used by the SVG; ANSI-256 for terminals)
# ---------------------------------------------------------------------------
PURPLE_LIGHT = (0xC0, 0x84, 0xFC)  # #C084FC
PURPLE_MID = (0x93, 0x33, 0xEA)    # #9333EA
PURPLE_DARK = (0x6D, 0x28, 0xD9)   # #6D28D9

# Classic ANSI-256 purple codes chosen for the CLI (kept for compat).
_PURPLE = "\x1b[38;5;141m"      # PURPLE
_DARK = "\x1b[38;5;57m"         # DARK_PURPLE
_BRIGHT = "\x1b[38;5;207m"      # BRIGHT_PURPLE
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"

# 256-colour purple ramp (light -> deep) bridging codes 207 -> 141 -> 57.
_PURPLE_RAMP: tuple[int, ...] = (207, 183, 177, 141, 99, 75, 57)


def _ansi256(index: int) -> str:
    return f"\x1b[38;5;{index}m"


def _rgb(r: int, g: int, b: int) -> str:
    """ANSI 24-bit foreground color."""
    return f"\x1b[38;2;{r};{g};{b}m"


def _mix(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    position: float,
) -> tuple[int, int, int]:
    """Interpolate between two RGB colors (clamped 0.0 -> start, 1.0 -> end)."""
    position = max(0.0, min(1.0, position))
    return tuple(
        round(a + (b - a) * position) for a, b in zip(start, end)
    )


def _ramp_index(step: int, total: int) -> int:
    """Map a column step in [0, total] onto the ANSI-256 purple ramp."""
    if total <= 1:
        pos = 0.0
    else:
        pos = step / (total - 1)
    idx = int(pos * (len(_PURPLE_RAMP) - 1))
    return _PURPLE_RAMP[min(idx, len(_PURPLE_RAMP) - 1)]


# ---------------------------------------------------------------------------
# Terminal capability
# ---------------------------------------------------------------------------
def _supports_color(force: Optional[bool] = None, stream: Optional[TextIO] = None) -> bool:
    """Color availability given environment / stream policy."""
    if force is True:
        return True
    if force is False:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover - exotic streams
        return False


def _can_unicode() -> bool:
    """True when the stream can encode the banner's glyph set."""
    try:
        enc = sys.stdout.encoding or "utf-8"
        "\u2588\u2550\u2554\u2557\u255a\u255d\u2022".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def _cell_char() -> str:
    """Return the filled cell char the current stream can actually encode.

    UTF-8-capable streams get the solid block; on encodings that cannot
    represent it we degrade to ``#`` (which still matches the block look).
    """
    try:
        enc = sys.stdout.encoding or "utf-8"
        "\u2588".encode(enc)
        return "\u2588"
    except (LookupError, UnicodeEncodeError):
        return "#"


# ---------------------------------------------------------------------------
# Wordmark geometry (FIAE brand letterforms, 5 cols x 6 rows each)
# ---------------------------------------------------------------------------
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


def _build_wordmark_rows() -> list[list[bool]]:
    """One logical bitmap for FIAE: single spacer column between letters."""
    letters = [_LETTERS[ch] for ch in PRODUCT_NAME]
    height = len(letters[0])
    rows: list[list[bool]] = []
    for r in range(height):
        row: list[bool] = []
        for i, glyph in enumerate(letters):
            if i:
                row.append(False)
            row.extend(cell == "#" for cell in glyph[r])
        rows.append(row)
    return rows


def _render_letter(letter: str, cell: Optional[str] = None) -> list[str]:
    """Expand a 5x6 glyph pattern into rendered rows."""
    if cell is None:
        cell = _cell_char()
    return [row.replace("#", cell).replace(".", " ") for row in _LETTERS[letter]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def banner(force_color: Optional[bool] = None) -> str:
    """Return the ``FIAE`` wordmark with a per-column purple gradient.

    Color flows continuously left -> right across the whole wordmark
    (light #C084FC family -> deep #6D28D9 family), mirroring the horizontal
    gradient of ``logo.svg``. Rendered on the ANSI-256 ramp for terminal
    compatibility.
    """
    glyphs = [_render_letter(ch) for ch in PRODUCT_NAME]
    height = len(glyphs[0])
    chars_per_letter = 5
    spacing = 1
    total_cols = chars_per_letter * len(glyphs) + spacing * (len(glyphs) - 1)

    if not _supports_color(force_color):
        lines = []
        for r in range(height):
            parts = []
            for gi, g in enumerate(glyphs):
                parts.append(g[r])
                if gi < len(glyphs) - 1:
                    parts.append(" ")
            lines.append("".join(parts))
        return "\n".join(lines)

    lines = []
    for r in range(height):
        out: list[str] = []
        col = 0
        for gi, g in enumerate(glyphs):
            for ch in g[r]:
                if ch != " ":
                    out.append(_ansi256(_ramp_index(col, total_cols)))
                    out.append(ch)
                else:
                    out.append(" ")
                col += 1
            if gi < len(glyphs) - 1:
                out.append(" ")
                col += 1
        lines.append("".join(out) + _RESET)
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
    return "\n".join([*painted, _RESET])


def compact_mark(force_color: Optional[bool] = None) -> str:
    """Terminal-native version of the FIAE graphical symbol.

        >--o
           |
           o
           |
           o

    Represents: terminal input -> feature nodes -> architecture.
    ASCII-safe (renders on any console); falls back to plain text when
    the stream cannot encode the glyphs or color is off.
    """
    lines = (
        ">--o",
        "   |",
        "   o",
        "   |",
        "   o",
    )
    if not _supports_color(force_color):
        return "\n".join(lines)
    colors = (
        _ramp_index(0, 4),
        _ramp_index(1, 4),
        _ramp_index(2, 4),
        _ramp_index(3, 4),
        _ramp_index(4, 4),
    )
    rendered = [
        f"{_ansi256(rgb)}{line}{_RESET}" for line, rgb in zip(lines, colors)
    ]
    return "\n".join(rendered)


def version_line() -> str:
    """One-line brand + version, ideal for ``fiae --version`` / prompt."""
    from . import __version__  # type: ignore[attr-defined]
    return f"fiae {__version__} - Feature Intelligence & Architecture Engine"


# ---------------------------------------------------------------------------
# Figlet banner (lowercase "fiae" + bordered tagline box)
# ---------------------------------------------------------------------------
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
        logo = "\n".join(f"{_PURPLE}{line}{_RESET}" for line in _FIGLET_LOGO.splitlines())
    else:
        logo = _FIGLET_LOGO

    if not with_box:
        return logo

    box_w = 52 if width is None else max(40, width)

    # NOTE: bar must be built outside the f-string expressions — backslash
    # escapes inside f-string expression parts are a SyntaxError on 3.10/3.11.
    bar = "\u2550" * box_w

    def top() -> str:
        return f"{_DARK}\u2554{bar}\u2557{_RESET}" if color else f"\u2554{bar}\u2557"

    def bot() -> str:
        return f"{_DARK}\u255a{bar}\u255d{_RESET}" if color else f"\u255a{bar}\u255d"

    def row(text: str, dim: bool = False) -> str:
        pad = box_w - 2 - len(text)
        if pad < 0:
            text = text[: box_w - 3] + "\u2026"
            pad = 0
        body = f" {text}{' ' * pad} "
        if color:
            inner = f"{_DARK if dim else _PURPLE}{body}{_RESET}"
            return f"{_DARK}\u2551{_RESET}{inner}{_DARK}\u2551{_RESET}"
        return f"\u2551{body}\u2551"

    title = "FEATURE INTELLIGENCE & ARCHITECTURE ENGINE"
    caption = f"fiae {__version__}  \u2022  {PRODUCT_TAGLINE}"

    return "\n".join([logo, top(), row(title), row(caption, dim=True), bot()])


def render_banner(tagline: bool = True, force_color: Optional[bool] = None) -> str:
    """Complete FIAE brand banner: wordmark + long name + tagline.

    Returns the string; never prints (deterministic, testable,
    redirect-safe).
    """
    parts = [banner(force_color=force_color)]
    if tagline:
        if _supports_color(force_color):
            parts.append("")
            parts.append(f"{_BOLD}{_PURPLE}{PRODUCT_LONG_NAME}{_RESET}")
            parts.append(f"{_DIM}{_ansi256(_PURPLE_RAMP[1])}{PRODUCT_TAGLINE}{_RESET}")
        else:
            parts.append("")
            parts.append(PRODUCT_LONG_NAME)
            parts.append(PRODUCT_TAGLINE)
    return "\n".join(parts)


def splash(tagline: bool = True, force_color: Optional[bool] = None) -> str:
    """Full CLI splash: gradient wordmark + long name + tagline."""
    if _can_unicode() and tagline:
        # Rich terminals get the figlet + bordered box presentation.
        return figlet(force_color=force_color, with_box=True)
    return render_banner(tagline=tagline, force_color=force_color)


__all__ = [
    "PRODUCT_LONG_NAME",
    "PRODUCT_NAME",
    "PRODUCT_TAGLINE",
    "PURPLE_DARK",
    "PURPLE_LIGHT",
    "PURPLE_MID",
    "banner",
    "compact_mark",
    "figlet",
    "pixel_f",
    "render_banner",
    "splash",
    "version_line",
]
