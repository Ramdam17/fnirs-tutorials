"""Tests of the house style: contrast of its colours and what ``apply_style`` sets."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib as mpl
import pytest

from fnirs_tutorials.style import PALETTE, STYLE, apply_style

# WCAG 2.1 success criterion 1.4.11 (Non-text Contrast): graphical objects, 3 to 1.
MIN_GRAPHIC_CONTRAST = 3.0
# WCAG 2.1 success criterion 1.4.3 (Contrast, Minimum): text, 4.5 to 1.
MIN_TEXT_CONTRAST = 4.5

FACE_KEYS = ["axes.facecolor", "figure.facecolor"]
TEXT_KEYS = [
    "text.color",
    "axes.labelcolor",
    "axes.titlecolor",
    "xtick.color",
    "ytick.color",
]


def relative_luminance(hex_color: str) -> float:
    """Relative luminance of a colour, as defined by WCAG 2.1.

    Parameters
    ----------
    hex_color : str
        Colour as ``"#rrggbb"``.

    Returns
    -------
    float
        ``L = 0.2126 R + 0.7152 G + 0.0722 B`` in [0, 1], where each sRGB channel
        ``c`` in [0, 1] is linearised as ``c / 12.92`` if ``c <= 0.04045`` and
        ``((c + 0.055) / 1.055) ** 2.4`` otherwise.

    Notes
    -----
    Source: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance

    Examples
    --------
    >>> relative_luminance("#ffffff")
    1.0
    """
    digits = hex_color.lstrip("#")
    channels = [int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(color_a: str, color_b: str) -> float:
    """Contrast ratio of two colours, as defined by WCAG 2.1.

    Parameters
    ----------
    color_a, color_b : str
        Colours as ``"#rrggbb"``, in any order.

    Returns
    -------
    float
        ``(L1 + 0.05) / (L2 + 0.05)``, with L1 the luminance of the lighter colour
        and L2 that of the darker one; between 1 and 21.

    Notes
    -----
    Source: https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio

    Examples
    --------
    >>> contrast_ratio("#000000", "#ffffff")
    21.0
    """
    lum_a, lum_b = relative_luminance(color_a), relative_luminance(color_b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.fixture
def restore_rcparams() -> Iterator[None]:
    """Leave no matplotlib global state behind after a test."""
    with mpl.rc_context():
        yield


@pytest.mark.parametrize("face_key", FACE_KEYS)
@pytest.mark.parametrize("color", PALETTE)
def test_palette_colour_contrast(color: str, face_key: str) -> None:
    ratio = contrast_ratio(color, str(STYLE[face_key]))
    assert ratio >= MIN_GRAPHIC_CONTRAST, f"{color} on {face_key}: {ratio:.2f}"


@pytest.mark.parametrize("face_key", FACE_KEYS)
@pytest.mark.parametrize("text_key", TEXT_KEYS)
def test_text_colour_contrast(text_key: str, face_key: str) -> None:
    ratio = contrast_ratio(str(STYLE[text_key]), str(STYLE[face_key]))
    assert ratio >= MIN_TEXT_CONTRAST, f"{text_key} on {face_key}: {ratio:.2f}"


def test_contrast_helper_known_values() -> None:
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert contrast_ratio("#7c3aed", "#7c3aed") == pytest.approx(1.0)


def test_palette_has_nine_distinct_colours() -> None:
    assert len(PALETTE) == 9
    assert len({c.lower() for c in PALETTE}) == 9


def test_apply_style_sets_rcparams_and_colour_cycle(restore_rcparams: None) -> None:
    apply_style()
    for key, value in STYLE.items():
        # RcParams validates and normalises on assignment (e.g. a family string
        # becomes a list), so the expected value goes through the same validation.
        expected = mpl.RcParams({key: value})[key]
        assert mpl.rcParams[key] == expected, key
    cycle = [entry["color"] for entry in mpl.rcParams["axes.prop_cycle"]]
    assert cycle == PALETTE
