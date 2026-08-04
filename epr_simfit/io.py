"""Robust text/ASC import for cw-EPR spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_FIELD_RANGE_MT
from .metadata_parser import parse_header


@dataclass
class EPRParseResult:
    dataframe: pd.DataFrame
    header_text: str
    metadata: dict[str, Any]
    detected_field_unit: str
    original_field_unit: str
    numeric_rows: int
    detected_columns: int
    warnings: list[str]
    filename: str | None = None
    field_col: int = 0
    intensity_col: int = 1


def _float_tokens(line: str) -> list[float] | None:
    line = line.strip()
    if not line:
        return None
    normalized = line.replace(",", " ").replace(";", " ").replace("\t", " ")
    pieces = [p for p in normalized.split() if p]
    if not pieces:
        return None
    vals = []
    for piece in pieces:
        try:
            vals.append(float(piece))
        except ValueError:
            return None
    return vals


def split_header_numeric(text: str) -> tuple[str, list[list[float]]]:
    header: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        tokens = _float_tokens(line)
        if tokens is None:
            header.append(line)
        else:
            rows.append(tokens)
    return "\n".join(header), rows


def _unit_from_header(metadata: dict[str, Any]) -> str | None:
    unit = str(metadata.get("field_unit", "")).lower()
    if "gauss" in unit or unit == "g":
        return "Gauss"
    if "mt" in unit:
        return "mT"
    return None


def detect_field_unit(field: np.ndarray, metadata: dict[str, Any], unit: str = "Auto") -> str:
    if unit and unit.lower() != "auto":
        return "Gauss" if unit.lower().startswith("g") else "mT"
    header_unit = _unit_from_header(metadata)
    if header_unit:
        return header_unit
    median = float(np.nanmedian(np.abs(field))) if field.size else 0.0
    if median > 1000:
        return "Gauss"
    if 250 <= median <= 500:
        return "mT"
    return "mT"


def _axis_from_single_column(n: int, metadata: dict[str, Any]) -> np.ndarray:
    center = metadata.get("center_field")
    sweep = metadata.get("sweep_width")
    if center is not None and sweep is not None:
        start = float(center) - float(sweep) / 2.0
        end = float(center) + float(sweep) / 2.0
        return np.linspace(start, end, n)
    start, end = DEFAULT_FIELD_RANGE_MT
    return np.linspace(start, end, n)


def parse_epr_text(
    text: str,
    filename: str | None = None,
    field_col: int = 0,
    intensity_col: int = 1,
    field_unit: str = "Auto",
    mw_frequency_override: float | None = None,
) -> EPRParseResult:
    header, numeric = split_header_numeric(text)
    metadata = parse_header(header)
    warnings: list[str] = []

    if not numeric:
        raise ValueError("No numeric spectrum rows were detected.")

    max_cols = max(len(row) for row in numeric)
    rectangular = [row + [np.nan] * (max_cols - len(row)) for row in numeric]
    arr = np.asarray(rectangular, dtype=float)
    arr = arr[np.isfinite(arr).any(axis=1)]
    if arr.size == 0:
        raise ValueError("Numeric rows were found but all values were NaN/inf.")

    if max_cols == 1:
        intensity = arr[:, 0]
        field = _axis_from_single_column(len(intensity), metadata)
        warnings.append("Only one numeric column was detected; a field axis was generated.")
        field_col = 0
        intensity_col = 0
    else:
        if field_col >= max_cols or intensity_col >= max_cols:
            raise ValueError(f"Requested columns exceed detected column count ({max_cols}).")
        field = arr[:, field_col]
        intensity = arr[:, intensity_col]

    mask = np.isfinite(field) & np.isfinite(intensity)
    field = np.asarray(field[mask], dtype=float)
    intensity = np.asarray(intensity[mask], dtype=float)

    original_unit = detect_field_unit(field, metadata, field_unit)
    if original_unit == "Gauss":
        field_mt = field / 10.0
        detected = "Gauss -> mT"
    else:
        field_mt = field
        detected = "mT"

    order = np.argsort(field_mt)
    df = pd.DataFrame(
        {
            "Field_mT": field_mt[order],
            "Intensity_raw": intensity[order],
        }
    )
    if mw_frequency_override:
        metadata["microwave_frequency_GHz"] = float(mw_frequency_override)
    if "microwave_frequency_GHz" not in metadata:
        warnings.append("Microwave frequency was not detected; using the GUI/default value is recommended.")

    return EPRParseResult(
        dataframe=df,
        header_text=header,
        metadata=metadata,
        detected_field_unit=detected,
        original_field_unit=original_unit,
        numeric_rows=len(df),
        detected_columns=max_cols,
        warnings=warnings,
        filename=filename,
        field_col=field_col,
        intensity_col=intensity_col,
    )


def read_epr_file(path: str | Path, **kwargs: Any) -> EPRParseResult:
    p = Path(path)
    return parse_epr_text(p.read_text(encoding="utf-8", errors="replace"), filename=p.name, **kwargs)


def uploaded_to_text(uploaded_file: Any) -> str:
    data = uploaded_file.read()
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")
