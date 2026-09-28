"""Tests for the CLI visual identity module (doc 10, NFR-002)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from fiae.identity import banner, pixel_f, splash, version_line


def _no_esc(s: str) -> bool:
    return "\x1b" not in s


@pytest.mark.parametrize("kind", ["banner", "pixel_f", "splash"])
def test_plain_has_no_ansi(kind):
    fn = {"banner": banner, "pixel_f": pixel_f, "splash": splash}[kind]
    assert _no_esc(fn(force_color=False))


def test_colored_has_ansi():
    out = banner(force_color=True)
    assert "\x1b[" in out
    assert "\x1b[0m" in out  # reset sentinel present


@pytest.mark.parametrize("kind", ["banner", "pixel_f", "splash"])
def test_force_color_output(kind):
    fn = {"banner": banner, "pixel_f": pixel_f, "splash": splash}[kind]
    assert "\x1b[" in fn(force_color=True)


def test_banner_shape():
    lines = banner(force_color=False).splitlines()
    assert len(lines) == 6
    # each letter is 5 cells wide with 1 space between the four letters
    assert all(len(line) == 5 * 4 + 3 for line in lines)
    # every row must be non-empty (all four glyphs present at every height)
    assert all(line.strip() for line in lines)


def test_pixel_f_shape():
    from fiae.identity import _cell_char
    cell = _cell_char()
    lines = pixel_f(force_color=False).splitlines()
    assert len(lines) == 6
    assert all(len(line) == 5 for line in lines)
    # top and third rows are the full F crossbars
    assert lines[0].count(cell) == 5
    assert lines[2].count(cell) == 5


def test_splash_contents():
    from fiae.identity import _cell_char
    cell = _cell_char()
    out = splash(force_color=False)
    assert cell in out
    assert "FEATURE INTELLIGENCE & ARCHITECTURE ENGINE" in out
    assert "fiae" in out


def test_version_line():
    v = version_line()
    assert v.startswith("fiae ")
    # matches the package version
    from fiae import __version__
    assert __version__ in v


def test_no_tagline():
    lines = banner(force_color=False).splitlines()
    assert len(lines) == 6  # wordmark only


def test_cell_char_valid():
    from fiae.identity import _cell_char
    c = _cell_char()
    assert c in ("\u2588", "#")


def test_figlet_logo_lowercase():
    from fiae.identity import figlet, _can_unicode
    if not _can_unicode():
        pytest.skip("stream cannot encode figlet glyphs")
    out = figlet(force_color=False)
    assert "fiae" in out                 # lowercase wordmark in the caption
    assert "FEATURE INTELLIGENCE" in out
    assert out.count("\u2554") >= 1      # box top-left
    assert out.count("\u255d") >= 1      # box bottom-right


def test_figlet_no_box_is_logo_only():
    from fiae.identity import figlet, _can_unicode
    if not _can_unicode():
        pytest.skip("stream cannot encode figlet glyphs")
    out = figlet(force_color=False, with_box=False)
    assert "FEATURE INTELLIGENCE" not in out
    assert len(out.splitlines()) == 6    # logo rows only


def test_figlet_ascii_fallback():
    from fiae.identity import figlet, _can_unicode
    if not _can_unicode():
        # on a legacy console the fallback must still be a clean wordmark
        out = figlet(force_color=False, with_box=True)
        assert "\x1b" not in out and out.strip()


def test_brand_codes():
    """The banner must use the user's exact ANSI-256 purple codes."""
    from fiae.identity import _PURPLE, _DARK, _BRIGHT
    assert _PURPLE == "\x1b[38;5;141m"   # PURPLE
    assert _DARK == "\x1b[38;5;57m"      # DARK_PURPLE
    assert _BRIGHT == "\x1b[38;5;207m"   # BRIGHT_PURPLE


def test_banner_uses_ansi256_ramp():
    from fiae.identity import banner, _PURPLE_RAMP
    assert _PURPLE_RAMP[0] == 207 and _PURPLE_RAMP[-1] == 57
    out = banner(force_color=True)
    assert "38;5;141" in out or "38;5;57" in out or "38;5;207" in out


def test_cli_banner_force_color(capsys):
    """--color emits the ANSI-256 codes even when stdout is captured."""
    from fiae.cli import main
    assert main(["banner", "--color"]) == 0
    out = capsys.readouterr().out
    assert "38;5;141" in out


def test_cli_banner_no_color_wins(capsys):
    from fiae.cli import main
    assert main(["banner", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_cli_version_flag(capsys):
    from fiae.cli import main
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("fiae ")


def test_cli_banner_command(capsys):
    from fiae.cli import main
    assert main(["banner", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "FEATURE INTELLIGENCE & ARCHITECTURE ENGINE" in out


def test_cli_banner_no_tagline(capsys):
    from fiae.cli import main
    assert main(["banner", "--no-color", "--no-tagline"]) == 0
    out = capsys.readouterr().out
    assert "FEATURE INTELLIGENCE" not in out
    assert len(out.splitlines()) == 6


# ---------------------------------------------------------------------------
# Brand-system additions (M27): helpers, gradient, drift lock
# ---------------------------------------------------------------------------

def test_mix_endpoints_and_clamping():
    from fiae.identity import PURPLE_DARK, PURPLE_LIGHT, _mix
    assert _mix(PURPLE_LIGHT, PURPLE_DARK, 0.0) == PURPLE_LIGHT
    assert _mix(PURPLE_LIGHT, PURPLE_DARK, 1.0) == PURPLE_DARK
    # clamped outside [0, 1]
    assert _mix(PURPLE_LIGHT, PURPLE_DARK, -5.0) == PURPLE_LIGHT
    assert _mix(PURPLE_LIGHT, PURPLE_DARK, 5.0) == PURPLE_DARK


def test_wordmark_gradient_is_monotonic():
    """Left edge lighter than right edge: continuous brand gradient."""
    from fiae.identity import _PURPLE_RAMP, banner
    out = banner(force_color=True)
    first_code = _PURPLE_RAMP[0]
    last_code = _PURPLE_RAMP[-1]
    assert f"38;5;{first_code}" in out   # light purple at the left
    assert f"38;5;{last_code}" in out    # deep purple at the right


def test_wordmark_matches_letter_geometry():
    """Drift lock: plain banner must equal the _LETTERS grid exactly."""
    from fiae.identity import _cell_char, _LETTERS, banner
    cell = _cell_char()
    word = "FIAE"
    height = len(_LETTERS["F"])
    grid = []
    for r in range(height):
        row = ""
        for i, ch in enumerate(word):
            row += _LETTERS[ch][r].replace("#", cell).replace(".", " ")
            if i < len(word) - 1:
                row += " "
        grid.append(row)
    assert banner(force_color=False).splitlines() == grid


def test_render_banner_contents():
    from fiae.identity import PRODUCT_LONG_NAME, PRODUCT_TAGLINE, render_banner
    out = render_banner(force_color=False)
    assert PRODUCT_LONG_NAME in out
    assert PRODUCT_TAGLINE in out
    with_tag = render_banner(tagline=True, force_color=False).splitlines()
    without = render_banner(tagline=False, force_color=False).splitlines()
    assert len(with_tag) > len(without)


def test_compact_mark_shape():
    from fiae.identity import compact_mark
    lines = compact_mark(force_color=False).splitlines()
    assert len(lines) == 5
    assert lines[0].startswith(">")
    # no-color output must be plain
    assert "\x1b" not in compact_mark(force_color=False)
    assert "\x1b[" in compact_mark(force_color=True)


def test_no_color_env_suppresses_ansi(monkeypatch):
    import fiae.identity as ident
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b" not in ident.banner(force_color=None)
    assert "\x1b" not in ident.splash(force_color=None)


def test_brand_product_constants():
    from fiae.identity import PRODUCT_LONG_NAME, PRODUCT_NAME, PRODUCT_TAGLINE
    assert PRODUCT_NAME == "FIAE"
    assert "FEATURE INTELLIGENCE" in PRODUCT_LONG_NAME
    assert "Safety-first" in PRODUCT_TAGLINE
