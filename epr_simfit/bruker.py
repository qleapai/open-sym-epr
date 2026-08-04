"""Bruker EPR file readers: BES3T (.DTA/.DSC) and legacy ESP/WinEPR (.spc/.par).

BES3T is the modern Bruker Xepr/Elexsys format: an ASCII descriptor (``.DSC``)
plus a binary data file (``.DTA``).  The descriptor gives the number of points,
the abscissa range, the byte order, the number format, and the microwave
frequency; the data file holds the ordinate values as raw binary.

This reader returns the field axis (converted to mT), the intensity, and a
metadata dictionary (microwave frequency in GHz, title, units, etc.).
"""

from __future__ import annotations

import re

import numpy as np

# BES3T number formats -> numpy dtypes
_IRFMT = {"D": "f8", "F": "f4", "I": "i4", "S": "i2", "C": "i1"}


def parse_dsc(dsc_text: str) -> dict:
    """Parse a BES3T ``.DSC`` descriptor into a flat key -> value dict."""
    params: dict[str, str] = {}
    for line in dsc_text.splitlines():
        line = line.rstrip()
        if not line or line.startswith(("#", "*", ".DVC", "*\t")):
            continue
        # KEY<whitespace>VALUE  (value may contain spaces / quotes)
        m = re.match(r"^([A-Za-z0-9_]+)\s+(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key not in params:  # first occurrence (DESC layer) wins for axis keys
            params[key] = val
        else:
            params.setdefault("_dsl_" + key, val)
    return params


def _axis(xmin: float, xwid: float, xpts: int) -> np.ndarray:
    if xpts <= 1:
        return np.array([xmin], dtype=float)
    return xmin + np.arange(xpts, dtype=float) * (xwid / (xpts - 1))


def load_bes3t(dsc_text: str, dta_bytes: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a BES3T spectrum from descriptor text and data bytes.

    Returns (field_mT, intensity, metadata).  Handles 1D real or complex data;
    for complex data the magnitude is returned (real part also kept in metadata).
    """
    p = parse_dsc(dsc_text)

    xpts = int(float(p.get("XPTS", "0")))
    if xpts <= 0:
        raise ValueError("DSC missing or invalid XPTS.")
    xmin = float(p.get("XMIN", "0"))
    xwid = float(p.get("XWID", str(xpts - 1)))
    xuni = p.get("XUNI", "G").strip().strip("'\"")
    bseq = p.get("BSEQ", "BIG").strip().upper()
    irfmt = p.get("IRFMT", "D").strip().upper()
    ikkf = p.get("IKKF", "REAL").strip().upper()
    ypts = int(float(p.get("YPTS", "1"))) if p.get("YTYP", "NODATA") != "NODATA" else 1

    dtype = _IRFMT.get(irfmt, "f8")
    endian = ">" if bseq == "BIG" else "<"
    np_dtype = np.dtype(endian + dtype)

    raw = np.frombuffer(dta_bytes, dtype=np_dtype)
    complex_data = ikkf == "CPLX"
    if complex_data:
        raw = raw[0::2] + 1j * raw[1::2]

    # 2D handling: reshape to (ypts, xpts) and take the first trace if needed.
    total = xpts * max(ypts, 1)
    if raw.size >= total and ypts > 1:
        raw = raw.reshape(ypts, xpts)
        intensity_2d = raw
        intensity = np.asarray(raw[0])
    else:
        intensity = raw[:xpts]
        intensity_2d = None

    field = _axis(xmin, xwid, xpts)
    # Gauss -> mT (1 mT = 10 G)
    if xuni in ("G", "Gauss", "gauss"):
        field_mT = field / 10.0
        field_unit = "Gauss -> mT"
    elif xuni in ("mT", "mt"):
        field_mT = field
        field_unit = "mT"
    else:
        # Heuristic: large values are Gauss
        field_mT = field / 10.0 if np.median(np.abs(field)) > 1000 else field
        field_unit = f"{xuni} (assumed)"

    mwfq_hz = float(p.get("MWFQ", "0") or 0)
    metadata = {
        "title": p.get("TITL", "").strip().strip("'\""),
        "microwave_frequency_GHz": mwfq_hz / 1e9 if mwfq_hz else None,
        "n_points": xpts,
        "field_unit_original": xuni,
        "detected_field_unit": field_unit,
        "byte_order": "big-endian" if bseq == "BIG" else "little-endian",
        "number_format": irfmt,
        "complex": complex_data,
        "center_field_mT": float(p.get("A1CT", "0")) * 1000 if p.get("A1CT") else None,
        "modulation_amplitude_mT": float(p.get("B0MA", "0")) * 1000 if p.get("B0MA") else None,
        "power_mW": float(p.get("MWPW", "0")) * 1000 if p.get("MWPW") else None,
        "source": "Bruker BES3T (.DTA/.DSC)",
    }
    out_intensity = np.abs(intensity) if complex_data else np.real(intensity).astype(float)
    if intensity_2d is not None:
        metadata["n_slices"] = int(ypts)
    return field_mT, out_intensity, metadata


def bruker_to_text(field_mT: np.ndarray, intensity: np.ndarray, metadata: dict) -> str:
    """Render a parsed Bruker spectrum as two-column ASC text (Field_mT, Intensity).

    Lets the standard text-parsing pipeline ingest Bruker BES3T data unchanged.
    """
    mw = metadata.get("microwave_frequency_GHz")
    header = [
        f"# Sample: {metadata.get('title', 'Bruker import')}",
        f"# Source: {metadata.get('source', 'Bruker BES3T')}",
        f"# Microwave Frequency: {mw:.6f} GHz" if mw else "# Microwave Frequency: unknown",
        "# Field unit: mT",
        "Field_mT\tIntensity",
    ]
    rows = [f"{b:.6f}\t{v:.9g}" for b, v in zip(field_mT, intensity)]
    return "\n".join(header + rows) + "\n"


def load_bes3t_files(dsc_path: str, dta_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convenience: load a BES3T pair from file paths."""
    from pathlib import Path
    dsc_text = Path(dsc_path).read_text(encoding="latin-1")
    dta_bytes = Path(dta_path).read_bytes()
    return load_bes3t(dsc_text, dta_bytes)
