"""Header parsing for Bruker/ASCII cw-EPR files."""

from __future__ import annotations

import re
from typing import Any


def _first_float(pattern: str, text: str, flags: int = re.IGNORECASE) -> float | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _line_value(label_regex: str, header: str) -> str | None:
    for line in header.splitlines():
        if re.search(label_regex, line, re.IGNORECASE):
            cleaned = line.lstrip("#; ").strip()
            parts = re.split(r"[:=]", cleaned, maxsplit=1)
            if len(parts) == 2:
                return parts[1].strip()
            return cleaned
    return None


def parse_header(header: str) -> dict[str, Any]:
    """Extract common EPR metadata from free-form header text."""
    text = header or ""
    meta: dict[str, Any] = {"header_line_count": len(text.splitlines())}

    freq = _first_float(r"(?:MWFQ|Microwave\s+Frequency)\D*([0-9]+(?:\.[0-9]+)?)\s*(GHz|MHz)?", text)
    if freq is None:
        freq = _first_float(r"\b([0-9]+(?:\.[0-9]+)?)\s*GHz\b", text)
    if freq is not None:
        mhz_match = re.search(r"(?:MWFQ|Microwave\s+Frequency)\D*[0-9.]+\s*MHz", text, re.IGNORECASE)
        meta["microwave_frequency_GHz"] = freq / 1000.0 if mhz_match else freq

    field_unit = _line_value(r"\bfield\s*unit\b|\bunit\b", text)
    if field_unit:
        if re.search(r"\bG\b|Gauss", field_unit, re.IGNORECASE):
            meta["field_unit"] = "Gauss"
        elif re.search(r"\bmT\b|milli", field_unit, re.IGNORECASE):
            meta["field_unit"] = "mT"

    patterns = {
        "modulation_amplitude": r"(?:ModAmp|Modulation\s+Amplitude|Modulation)\D*([0-9]+(?:\.[0-9]+)?)",
        "modulation_frequency": r"(?:Modulation\s+Frequency)\D*([0-9]+(?:\.[0-9]+)?)",
        "power": r"(?:Power)\D*([0-9]+(?:\.[0-9]+)?)",
        "temperature": r"(?:Temperature|Temp)\D*([0-9]+(?:\.[0-9]+)?)",
        "number_of_points": r"(?:XPTS|YPTS|Points|Number\s+of\s+points)\D*([0-9]+)",
        "center_field": r"(?:Center|Center\s+Field)\D*([0-9]+(?:\.[0-9]+)?)",
        "sweep_width": r"(?:Sweep|Sweep\s+Width|Range)\D*([0-9]+(?:\.[0-9]+)?)",
        "conversion_time": r"(?:Conversion\s+Time)\D*([0-9]+(?:\.[0-9]+)?)",
        "time_constant": r"(?:Time\s+Constant)\D*([0-9]+(?:\.[0-9]+)?)",
        "receiver_gain": r"(?:Receiver|Gain|Receiver\s+Gain)\D*([0-9]+(?:\.[0-9]+)?)",
    }
    for key, pattern in patterns.items():
        value = _first_float(pattern, text)
        if value is not None:
            meta[key] = value

    sample = _line_value(r"\bSample\b|\bSample\s+Name\b", text)
    if sample:
        meta["sample_name"] = sample
    condition = _line_value(r"\bCondition\b|\bExperiment\b", text)
    if condition:
        meta["condition"] = condition
    date = _line_value(r"\bDate\b|\bTime\b", text)
    if date:
        meta["date_time"] = date

    notes = []
    keywords = {
        "gas_N2": r"\bN2\b|nitrogen",
        "gas_Ar": r"\bAr\b|argon",
        "gas_air_o2": r"\bO2\b|air|oxygen",
        "light": r"\blight\b|illuminat|photo",
        "dark": r"\bdark\b",
        "dmso": r"\bDMSO\b",
        "water": r"\bwater\b|aqueous|H2O",
        "pbn": r"\bPBN\b",
        "pd_ov_bmo": r"Pd.*Ov.*Bi2MoO6|Pd.*oxygen.*vacancy|Pd-Ov-Bi2MoO6",
    }
    for key, pattern in keywords.items():
        if re.search(pattern, text, re.IGNORECASE):
            notes.append(key)
    meta["header_keywords"] = notes
    return meta


def metadata_table(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"metadata": key, "value": value} for key, value in metadata.items()]
