"""Matplotlib house style shared by all notebooks of the series.

This is an aesthetic choice (a house style: light canvas, airy layout, a vibrant
categorical palette), not a methodological one, so it carries no citation. The canvas
is light so that a figure reads on both modes of the book and in print. The palette is
the one of the lab's plotting standard, with four colours darkened one shade so that
every colour reaches a contrast ratio of 3 to 1 on white. That threshold is the one
that the Web Content Accessibility Guidelines 2.1 give for graphical objects (success
criterion 1.4.11, https://www.w3.org/TR/WCAG21/#non-text-contrast), and the 4.5 to 1
of criterion 1.4.3 (https://www.w3.org/TR/WCAG21/#contrast-minimum) is applied to the
text colours. Those guidelines are written for web content, so applying their
thresholds to figures is a choice of this project.
"""

from __future__ import annotations

import matplotlib as mpl

__all__ = ["PALETTE", "STYLE", "apply_style"]

# Categorical palette, in the order of the matplotlib colour cycle.
PALETTE: list[str] = [
    "#7c3aed",
    "#ec4899",
    "#d97706",  # replaces #f59e0b, too pale on white (2.15 to 1)
    "#059669",  # replaces #10b981, too pale on white (2.54 to 1)
    "#3b82f6",
    "#ef4444",
    "#8b5cf6",
    "#0891b2",  # replaces #06b6d4, too pale on white (2.43 to 1)
    "#65a30d",  # replaces #84cc16, too pale on white (1.98 to 1)
]

STYLE: dict[str, object] = {
    # Canvas
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    # Text
    "text.color": "#1f2328",
    "axes.labelcolor": "#1f2328",
    "xtick.color": "#656d76",
    "ytick.color": "#656d76",
    "axes.titlecolor": "#1f2328",
    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#d0d7de",
    # Grid
    "axes.grid": True,
    "grid.color": "#d8dee4",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    # Typography
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    # Lines
    "lines.linewidth": 2.2,
    "lines.solid_capstyle": "round",
    # Figures
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (14, 6),
}


def apply_style() -> None:
    """Apply the house style to ``matplotlib.rcParams`` and set the colour cycle.

    Returns
    -------
    None

    Notes
    -----
    When to use: once, in the first code cell of every notebook, before any figure.
    The call changes global matplotlib state for the running process.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from fnirs_tutorials.style import apply_style
    >>> apply_style()
    >>> matplotlib.rcParams["axes.grid"]
    True
    """
    mpl.rcParams.update(STYLE)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)
