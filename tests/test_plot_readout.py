"""Tests for the additive cursor-readout (g-value) on spectrum plots."""
import numpy as np

from epr_simfit import plotting


def test_g_values_match_resonance_condition():
    # g = h*nu/(muB*B); at 9.816 GHz a g=2.0 line sits near 350.6 mT
    B = np.array([350.6])
    g = plotting._g_values(B, 9.816)[0]
    assert abs(g - 2.0) < 0.01


def test_hover_template_modes():
    assert "Intensity" in plotting._hover_template("xy", has_g=True)
    assert "customdata" in plotting._hover_template("g", has_g=True)
    both = plotting._hover_template("both", has_g=True)
    assert "Intensity" in both and "customdata" in both
    # g requested but unavailable -> falls back to xy (never blank)
    assert "Intensity" in plotting._hover_template("g", has_g=False)


def test_spectrum_figure_attaches_g_customdata():
    field = np.linspace(344.0, 356.0, 50)
    fig = plotting.spectrum_figure(field, {"experimental": np.sin(field)},
                                   readout="both", mw_frequency_GHz=9.82)
    tr = fig.data[0]
    assert tr.customdata is not None
    assert "customdata" in tr.hovertemplate
    # xy-only mode attaches no g customdata
    fig2 = plotting.spectrum_figure(field, {"y": np.cos(field)}, readout="xy", mw_frequency_GHz=9.82)
    assert fig2.data[0].customdata is None
