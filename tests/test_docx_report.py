"""Tests for the publication-ready DOCX report."""
import io
import zipfile

import numpy as np

from epr_simfit import docx_report


def _entry():
    B = np.linspace(344.0, 356.0, 120)
    return {
        "name": "Test mixture", "kind": "mixture", "field_mT": B,
        "experimental": np.sin(B), "total": np.sin(B) * 0.9,
        "curves": {"PBN-OH": np.sin(B) * 0.5, "PBN-CH3": np.sin(B) * 0.4},
        "rows": [
            {"component": "PBN-OH", "assignment": "PBN-OH", "g": 2.0056,
             "avals": [("14N", "N", 1.52), ("1H", "beta H", 0.28)],
             "linewidth_mT": 0.08, "eta": 0.5, "fraction_pct": 60.0},
            {"component": "PBN-CH3", "assignment": "PBN-CH3", "g": 2.0056,
             "avals": [("14N", "N", 1.55)], "linewidth_mT": 0.09, "eta": 0.5, "fraction_pct": 40.0},
        ],
        "R2": 0.89, "nrmse": 0.11, "aic": -100.0, "bic": -90.0, "n_params": 6,
        "methods_text": "The spectrum was decomposed by bounded least-squares.",
    }


def test_docx_is_valid_and_has_content():
    data = docx_report.build_report_docx([_entry()], meta={"software": "Open-Sym-EPR"})
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert "word/document.xml" in zf.namelist()
    xml = zf.read("word/document.xml").decode("utf-8")
    assert "Figure 1." in xml and "Table 1." in xml
    assert "PBN-OH" in xml and "Assignment" in xml
    # an embedded PNG figure exists
    assert any(n.startswith("word/media/") for n in zf.namelist())


def test_iso_superscript_and_avals():
    assert docx_report._iso_label("14N") == "¹⁴N"
    assert docx_report._avals_text([("14N", "N", 1.52)]) == "¹⁴N 15.2"


def test_multiple_entries_number_sequentially():
    e2 = _entry(); e2["name"] = "Second"
    data = docx_report.build_report_docx([_entry(), e2])
    xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8")
    assert "Figure 1." in xml and "Figure 2." in xml
    assert "Table 1." in xml and "Table 2." in xml
