"""Matplotlib house style shared by all notebooks of the series.

This is an aesthetic choice (a house style: dark canvas, airy layout, a vibrant
categorical palette), not a methodological one, so it carries no citation.
"""

from __future__ import annotations

import matplotlib as mpl

__all__ = ["PALETTE", "STYLE", "apply_style"]

# Categorical palette, in the order of the matplotlib colour cycle.
PALETTE: list[str] = [
    "#7c3aed",
    "#ec4899",
    "#f59e0b",
    "#10b981",
    "#3b82f6",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#84cc16",
]

STYLE: dict[str, object] = {
    # Canvas
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "savefig.facecolor": "#0d1117",
    # Text
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.titlecolor": "#ffffff",
    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#30363d",
    # Grid
    "axes.grid": True,
    "grid.color": "#21262d",
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
