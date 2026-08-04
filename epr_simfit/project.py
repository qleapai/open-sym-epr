"""Project save/load for Open-Sym-EPR — persist a working session and resume it later.

A project file is a single self-contained JSON document capturing the loaded
spectrum, the microwave frequency, preprocessing settings, any custom/selected
component models, and a summary of the latest fit. Loading it restores the
spectrum and settings so the user can continue where they left off.
"""
from __future__ import annotations

import json
import time

SCHEMA = "Open-Sym-EPR.project"
VERSION = "1.0"


def save_project(*, microwave_frequency_GHz: float,
                 spectrum_text: str | None = None,
                 filename: str | None = None,
                 preprocess: dict | None = None,
                 components_json: str | None = None,
                 fit_summary: dict | None = None,
                 config: dict | None = None,
                 notes: str = "") -> bytes:
    """Serialize the working session into a portable project (JSON bytes)."""
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "microwave_frequency_GHz": float(microwave_frequency_GHz),
        "spectrum_text": spectrum_text,      # original file text (2-column or ASC)
        "filename": filename,
        "preprocess": preprocess or {},
        "components_json": components_json,   # epr_simfit.user_models JSON of components
        "fit_summary": fit_summary or {},
        "config": config or {},
        "notes": notes,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def load_project(data: bytes | str) -> dict:
    """Parse a project file and return its payload dict; validates the schema."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", errors="replace")
    obj = json.loads(data)
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise ValueError("Not a Open-Sym-EPR project file (missing or wrong schema).")
    return obj


def project_summary(obj: dict) -> str:
    """One-line human summary of a loaded project for display."""
    n = "?"
    if obj.get("spectrum_text"):
        n = sum(1 for ln in obj["spectrum_text"].splitlines()
                if ln[:1].isdigit() or ln[:1] in "+-.")
    return (f"saved {obj.get('saved_utc', '?')} · ν = {obj.get('microwave_frequency_GHz', '?')} GHz · "
            f"{n} data rows · {'fit present' if obj.get('fit_summary') else 'no fit'}")
