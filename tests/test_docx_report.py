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


def test_species_runs_formats_subscripts_and_isotopes():
    r = dict(docx_report._species_runs("PBN-CH3"))
    assert r.get("3") == "sub"                       # formula subscript
    r2 = docx_report._species_runs("Nitroxide-14N")
    assert ("14", "sup") in r2                        # isotope superscript
    # ratio patterns like 1:2:2:1 are left normal
    assert all(s is None for _t, s in docx_report._species_runs("DMPO-OH (1:2:2:1)"))


def test_iso_superscript_and_avals():
    assert docx_report._iso_label("14N") == "¹⁴N"
    assert docx_report._avals_text([("14N", "N", 1.52)]) == "¹⁴N 15.2"


def test_detailed_parameter_tables_included():
    import pandas as pd
    e = _entry()
    e["param_df"] = pd.DataFrame({
        "Component": ["pbn_oh", "pbn_oh"], "Parameter": ["weight", "A_mT"],
        "Value": [0.6, 1.52], "Std. error": [0.01, 0.02],
        "Lower": [0.0, 1.3], "Upper": [None, 1.7], "Status": ["fitted", "fitted"]})
    e["mc_df"] = pd.DataFrame({
        "Component": ["pbn_oh"], "Parameter": ["A_mT"], "Value": [1.52],
        "MC σ": [0.03], "CI 2.5%": [1.46], "CI 97.5%": [1.58]})
    xml = zipfile.ZipFile(io.BytesIO(docx_report.build_report_docx([e]))).read(
        "word/document.xml").decode("utf-8")
    # summary Table 1, detailed params Table 2, Monte-Carlo Table 3
    assert "Table 1." in xml and "Table 2." in xml and "Table 3." in xml
    assert "Std. error" in xml and "MC" in xml
    assert "fitted spin-Hamiltonian parameters" in xml


def test_multiple_entries_number_sequentially():
    e2 = _entry(); e2["name"] = "Second"
    data = docx_report.build_report_docx([_entry(), e2])
    xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8")
    assert "Figure 1." in xml and "Figure 2." in xml
    assert "Table 1." in xml and "Table 2." in xml
