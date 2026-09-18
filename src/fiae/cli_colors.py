"""FIAE CLI color system — purple identity for every terminal touchpoint.

All CLI commands use this module for branded, consistent output:
- Headers with purple ANSI-256 gradient
- Status badges (PASS/FAIL/WARN)
- Table formatting
- Section dividers
- Error/success formatting

Respects NO_COLOR / FORCE_COLOR / TTY detection (doc 10, NFR-002).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ── Brand palette ───────────────────────────────────────────────────────
PURPLE = "\x1b[38;5;141m"       # Primary purple
DARK_PURPLE = "\x1b[38;5;57m"   # Dark purple (headers, borders)
BRIGHT_PURPLE = "\x1b[38;5;207m" # Bright purple (highlights)
GREEN = "\x1b[38;5;114m"        # Success green
RED = "\x1b[38;5;167m"          # Error red
YELLOW = "\x1b[38;5;186m"       # Warning yellow
CYAN = "\x1b[38;5;117m"         # Info cyan
GRAY = "\x1b[38;5;245m"         # Dimmed gray
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def _supports_color(force: Optional[bool] = None) -> bool:
    """Check if the terminal supports ANSI color."""
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
    except Exception:
        return False


# ── Color helpers ───────────────────────────────────────────────────────

def _c(text: str, color: str, force: Optional[bool] = None) -> str:
    """Wrap text in ANSI color if terminal supports it."""
    if not _supports_color(force):
        return text
    return f"{color}{text}{RESET}"


def purple(text: str, force: Optional[bool] = None) -> str:
    """Brand purple text."""
    return _c(text, PURPLE, force)


def dark(text: str, force: Optional[bool] = None) -> str:
    """Dark purple text."""
    return _c(text, DARK_PURPLE, force)


def bright(text: str, force: Optional[bool] = None) -> str:
    """Bright purple text."""
    return _c(text, BRIGHT_PURPLE, force)


def green(text: str, force: Optional[bool] = None) -> str:
    """Success green text."""
    return _c(text, GREEN, force)


def red(text: str, force: Optional[bool] = None) -> str:
    """Error red text."""
    return _c(text, RED, force)


def yellow(text: str, force: Optional[bool] = None) -> str:
    """Warning yellow text."""
    return _c(text, YELLOW, force)


def cyan(text: str, force: Optional[bool] = None) -> str:
    """Info cyan text."""
    return _c(text, CYAN, force)


def gray(text: str, force: Optional[bool] = None) -> str:
    """Dimmed gray text."""
    return _c(text, GRAY, force)


def bold(text: str, force: Optional[bool] = None) -> str:
    """Bold text."""
    return _c(text, BOLD, force)


# ── Formatted output helpers ────────────────────────────────────────────

def _can_unicode() -> bool:
    """Check if the terminal can encode box-drawing characters."""
    try:
        enc = sys.stdout.encoding or "utf-8"
        "═╔╗╚╝─│·•✓✗⚠●━─".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def header(text: str, force: Optional[bool] = None) -> str:
    """FIAE-branded header: ╔══ FIAE — {text} ══╗"""
    if not _supports_color(force):
        return f"=== FIAE — {text} ==="
    if not _can_unicode():
        return f"=== FIAE — {text} ==="
    w = 56
    inner = f" {text} "
    pad = max(0, w - len(text) - 2)
    left = pad // 2
    right = pad - left
    line = f"{'═' * left}{inner}{'═' * right}"
    return f"{DARK_PURPLE}╔{line}╗{RESET}"


def footer(force: Optional[bool] = None) -> str:
    """FIAE-branded footer line."""
    if not _supports_color(force):
        return "=" * 58
    if not _can_unicode():
        return "=" * 58
    return f"{DARK_PURPLE}╚{'═' * 56}╝{RESET}"


def section(text: str, force: Optional[bool] = None) -> str:
    """Section divider: ── {text} ──"""
    if not _supports_color(force):
        return f"--- {text} ---"
    if not _can_unicode():
        return f"--- {text} ---"
    w = 54
    inner = f" {text} "
    pad = max(0, w - len(text) - 2)
    left = pad // 2
    right = pad - left
    return f"{DARK_PURPLE}{'─' * left}{inner}{'─' * right}{RESET}"


def divider(force: Optional[bool] = None) -> str:
    """Simple horizontal divider."""
    if not _supports_color(force):
        return "-" * 58
    if not _can_unicode():
        return "-" * 58
    return f"{DARK_PURPLE}{'─' * 58}{RESET}"


def badge_pass(force: Optional[bool] = None) -> str:
    """PASS badge."""
    return _c(" PASS ", f"{BOLD}{GREEN}", force)


def badge_fail(force: Optional[bool] = None) -> str:
    """FAIL badge."""
    return _c(" FAIL ", f"{BOLD}{RED}", force)


def badge_warn(force: Optional[bool] = None) -> str:
    """WARN badge."""
    return _c(" WARN ", f"{BOLD}{YELLOW}", force)


def badge_info(force: Optional[bool] = None) -> str:
    """INFO badge."""
    return _c(" INFO ", f"{BOLD}{CYAN}", force)


def badge_skip(force: Optional[bool] = None) -> str:
    """SKIP badge."""
    return _c(" SKIP ", f"{BOLD}{GRAY}", force)


# ── Table helpers ───────────────────────────────────────────────────────

def key_value(key: str, value: str, force: Optional[bool] = None) -> str:
    """Format a key-value pair: key:  value (with purple key)."""
    if not _supports_color(force):
        return f"{key}:  {value}"
    return f"{PURPLE}{key}:{RESET}  {value}"


def table_row(*cells: str, widths: Optional[list[int]] = None,
              force: Optional[bool] = None) -> str:
    """Format a table row with optional fixed widths."""
    if widths is None:
        return "  ".join(cells)
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(f"{cell:<{w}}")
    return "  ".join(parts)


def table_header(*cells: str, widths: Optional[list[int]] = None,
                 force: Optional[bool] = None) -> str:
    """Format a table header row (dimmed)."""
    row = table_row(*cells, widths=widths)
    if not _supports_color(force):
        return row
    return f"{DIM}{row}{RESET}"


def table_separator(widths: list[int], force: Optional[bool] = None) -> str:
    """Table separator line."""
    if not _supports_color(force) or not _can_unicode():
        return "--".join("-" * w for w in widths)
    parts = ["─" * w for w in widths]
    return f"{DARK_PURPLE}{'──'.join(parts)}{RESET}"


# ── Progress helpers ────────────────────────────────────────────────────

def progress_step(current: int, total: int, label: str,
                  force: Optional[bool] = None) -> str:
    """Show progress: [ 3/10 ] Label"""
    counter = f"[{current:>{len(str(total))}}/{total}]"
    if not _supports_color(force):
        return f"  {counter} {label}"
    return f"  {DARK_PURPLE}{counter}{RESET} {label}"


def status_line(label: str, status: str, force: Optional[bool] = None) -> str:
    """Status line: Label ··· PASS/FAIL"""
    if not _supports_color(force):
        return f"  {label} ... {status}"
    sep = "..." if not _can_unicode() else "···"
    if status in ("PASS", "passed", "ok"):
        s = badge_pass(force)
    elif status in ("FAIL", "failed", "error"):
        s = badge_fail(force)
    elif status in ("WARN", "warning"):
        s = badge_warn(force)
    else:
        s = _c(status, PURPLE, force)
    return f"  {label} {DARK_PURPLE}{sep}{RESET} {s}"


# ── FIAE branding ──────────────────────────────────────────────────────

def version_tag(force: Optional[bool] = None) -> str:
    """FIAE version tag: fiae v0.0.1"""
    from . import __version__
    return f"{PURPLE}fiae{RESET} {_c(__version__, GRAY, force)}"


def brand_line(force: Optional[bool] = None) -> str:
    """Single-line brand: fiae v0.0.1 — Feature Intelligence & Architecture Engine"""
    from . import __version__
    if not _supports_color(force):
        return f"fiae {__version__} — Feature Intelligence & Architecture Engine"
    return (
        f"{PURPLE}fiae{RESET} "
        f"{GRAY}{__version__}{RESET} "
        f"{DARK_PURPLE}—{RESET} "
        f"Feature Intelligence & Architecture Engine"
    )


def command_header(command: str, description: str = "",
                   force: Optional[bool] = None) -> str:
    """Branded command header for every CLI subcommand."""
    from . import __version__
    if not _supports_color(force):
        lines = [
            f"fiae {__version__} -- Feature Intelligence & Architecture Engine",
            f"{'=' * 58}",
        ]
        if description:
            lines.append(description)
        return "\n".join(lines)
    if not _can_unicode():
        lines = [
            f"{PURPLE}fiae{RESET} {GRAY}{__version__}{RESET}  "
            f"{DARK_PURPLE}--{RESET}  Feature Intelligence & Architecture Engine",
            f"{DARK_PURPLE}{'=' * 58}{RESET}",
        ]
        if description:
            lines.append(f"  {description}")
        return "\n".join(lines)
    lines = [
        f"{PURPLE}fiae{RESET} {GRAY}{__version__}{RESET}  "
        f"{DARK_PURPLE}─{RESET}  Feature Intelligence & Architecture Engine",
        f"{DARK_PURPLE}{'━' * 58}{RESET}",
    ]
    if description:
        lines.append(f"  {description}")
    return "\n".join(lines)


def error_header(force: Optional[bool] = None) -> str:
    """Error header with FIAE branding."""
    if not _supports_color(force):
        return "ERROR"
    return f"{RED}{BOLD}ERROR{RESET} {DARK_PURPLE}fiae{RESET}"
