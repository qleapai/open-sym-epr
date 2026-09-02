"""Tests for per-plot CSV / PNG / SVG export."""
import numpy as np

from epr_simfit import fig_export
from epr_simfit.plotting import spectrum_figure, comparison_bar_figure
import pandas as pd


def test_csv_from_shared_x_spectrum():
    field = np.linspace(344.0, 356.0, 40)
    fig = spectrum_figure(field, {"experimental": np.sin(field), "fit": np.cos(field)})
    csv = fig_export.fig_to_csv(fig)
    header = csv.splitlines()[0].split(",")
    assert header[0] == "Magnetic field / mT"
    assert "experimental" in header and "fit" in header
    assert len(csv.splitlines()) == 41            # header + 40 points


def test_png_and_svg_bytes():
    field = np.linspace(0, 10, 30)
    fig = spectrum_figure(field, {"y": np.sin(field)})
    png = fig_export.fig_to_image(fig, "png")
    svg = fig_export.fig_to_image(fig, "svg")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert svg[:5] == b"<?xml" or b"<svg" in svg[:200]


def test_bar_figure_exports():
    df = pd.DataFrame({"model": ["A", "B"], "BIC": [10.0, 12.0]})
    fig = comparison_bar_figure(df, "BIC")
    csv = fig_export.fig_to_csv(fig)
    assert "A" in csv and "B" in csv
    png = fig_export.fig_to_image(fig, "png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
