"""Write canonical PDF and PNG exports for a figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_figure(
    figure: plt.Figure,
    stem: str,
    output_dir: Path,
    *,
    dpi: int,
    **savefig_options: Any,
) -> None:
    """Save a figure as PDF and high-resolution PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", **savefig_options}
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    figure.savefig(pdf, **options)
    figure.savefig(png, dpi=dpi, **options)
