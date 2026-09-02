"""Tests for the publication export bundle."""
import io
import zipfile

import numpy as np

from epr_simfit import pub_export


def _payload_manual():
    B = np.linspace(344.0, 356.0, 120)
    return {"kind": "manual", "field_mT": B, "experimental": np.sin(B), "total": np.sin(B) * 0.9,
            "components": {"PBN-OH": np.sin(B) * 0.5, "PBN-CH3": np.sin(B) * 0.4},
            "params_df": None}


def test_zip_contains_expected_files():
    items = {"Manual mixture": _payload_manual()}
    data = pub_export.build_selection_zip(items, meta={"software": "Open-Sym-EPR"})
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    assert "README.txt" in names
    assert "Manual_mixture/data.csv" in names
    assert "Manual_mixture/figure_overlay.png" in names
    assert "Manual_mixture/figure_decomposition.png" in names
    # data.csv has field + experimental + total + residual + 2 components
    csv = zf.read("Manual_mixture/data.csv").decode()
    header = csv.splitlines()[0]
    assert header.split(",") == ["Field_mT", "experimental", "total", "residual", "PBN-OH", "PBN-CH3"]


def test_png_is_valid_png():
    data = pub_export.build_selection_zip({"m": _payload_manual()})
    zf = zipfile.ZipFile(io.BytesIO(data))
    png = zf.read("m/figure_overlay.png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"          # PNG magic number


def test_safe_names_sanitised():
    assert pub_export._safe("PBN-OH / DMSO fit!") == "PBN-OH_DMSO_fit"
