"""Save, load, and share custom user-defined spin-component models.

Components are serialised to JSON via their Pydantic schema, so the full model
(isotropic and anisotropic fields: g-tensor, spin S, zero-field splitting,
per-nucleus hyperfine tensors) round-trips exactly.  User model packs can be
downloaded, shared, and re-imported across sessions or installations.
"""

from __future__ import annotations

import json
import csv
import io
from datetime import datetime, timezone
from pathlib import Path

from .spin_models import SpinComponent

SCHEMA_VERSION = "1.0"


def components_to_json(components: list[SpinComponent], name: str = "user_model") -> str:
    """Serialise a list of components into a portable JSON model pack."""
    payload = {
        "schema": "Open-Sym-EPR.user_model",
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_components": len(components),
        "components": [json.loads(c.model_dump_json()) for c in components],
    }
    return json.dumps(payload, indent=2)


def components_from_json(text: str) -> tuple[list[SpinComponent], dict]:
    """Parse a JSON model pack, returning (components, metadata).

    Accepts either a full model pack ({"components": [...]}) or a bare list of
    component dicts, for resilience to hand-edited files.
    """
    data = json.loads(text)
    if isinstance(data, dict) and "components" in data:
        raw = data["components"]
        meta = {k: v for k, v in data.items() if k != "components"}
    elif isinstance(data, list):
        raw = data
        meta = {}
    else:
        raise ValueError("Unrecognised model pack: expected an object with 'components' or a list.")
    components = [SpinComponent.model_validate(item) for item in raw]
    return components, meta


def _safe_id(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "custom_component"


def _float(row: dict, key: str, default: float) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _nuclei_from_text(text: str):
    from .spin_models import Nucleus

    nuclei = []
    for item in str(text or "").replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            isotope, value = item.split(":", 1)
        elif "=" in item:
            isotope, value = item.split("=", 1)
        else:
            continue
        isotope = isotope.strip()
        value = float(value.strip())
        nuclei.append(Nucleus(isotope=isotope, A_mT=value, label=isotope))
    return nuclei


def components_from_csv(text: str) -> tuple[list[SpinComponent], dict]:
    """Parse a flat CSV custom model library.

    Required column: ``name``. Useful optional columns:
    ``component_id, assignment, category, g, g_min, g_max, nuclei,
    linewidth_mT, lw_min, lw_max, eta, weight, spin_S, gx, gy, gz,
    D_MHz, E_MHz, mode``.

    Nuclei syntax: ``14N:1.55; 1H:0.30; 63Cu:8.5`` in mT.
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV model library has no header row.")
    components: list[SpinComponent] = []
    for row in reader:
        name = str(row.get("name") or row.get("display_name") or "").strip()
        if not name:
            continue
        cid = str(row.get("component_id") or ("custom_" + _safe_id(name))).strip()
        g = _float(row, "g", 2.003)
        gx = row.get("gx", "")
        gy = row.get("gy", "")
        gz = row.get("gz", "")
        g_tensor = None
        if gx not in ("", None) and gy not in ("", None) and gz not in ("", None):
            g_tensor = (float(gx), float(gy), float(gz))
            g = sum(g_tensor) / 3.0
        component = SpinComponent(
            component_id=cid,
            display_name=name,
            radical_assignment=str(row.get("assignment") or row.get("radical_assignment") or name),
            category=str(row.get("category") or "custom/uploaded"),
            g=g,
            g_bounds=(_float(row, "g_min", 1.90), _float(row, "g_max", 2.40)),
            nuclei=_nuclei_from_text(str(row.get("nuclei") or "")),
            linewidth_mT=_float(row, "linewidth_mT", 0.20),
            linewidth_bounds=(_float(row, "lw_min", 0.01), _float(row, "lw_max", 20.0)),
            eta=_float(row, "eta", 0.5),
            weight=_float(row, "weight", 0.3),
            interpretation=str(row.get("interpretation") or "Uploaded user-defined component."),
            warning=str(row.get("warning") or "Uploaded component: validate against references and controls."),
            spin_S=_float(row, "spin_S", 0.5),
            g_tensor=g_tensor,
            D_MHz=_float(row, "D_MHz", 0.0),
            E_MHz=_float(row, "E_MHz", 0.0),
            mode=str(row.get("mode") or "auto"),
        )
        components.append(component)
    if not components:
        raise ValueError("CSV model library did not contain any valid component rows.")
    return components, {"schema": "Open-Sym-EPR.user_model.csv", "n_components": len(components)}


def components_from_yaml(text: str) -> tuple[list[SpinComponent], dict]:
    """Parse YAML model packs with the same structure as JSON model packs."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ValueError("YAML model upload requires PyYAML.") from exc
    data = yaml.safe_load(text)
    if isinstance(data, dict) and "components" in data:
        raw = data["components"]
        meta = {k: v for k, v in data.items() if k != "components"}
    elif isinstance(data, list):
        raw = data
        meta = {}
    else:
        raise ValueError("Unrecognised YAML model pack: expected 'components' or a list.")
    return [SpinComponent.model_validate(item) for item in raw], meta


def components_from_text(text: str, filename: str = "uploaded_model.json") -> tuple[list[SpinComponent], dict]:
    """Load custom components from JSON, YAML, or CSV text based on filename."""
    lower = filename.lower()
    if lower.endswith(".csv"):
        return components_from_csv(text)
    if lower.endswith((".yaml", ".yml")):
        return components_from_yaml(text)
    return components_from_json(text)


def default_user_dir() -> Path:
    """Default directory for persisted user models (created on demand)."""
    d = Path.home() / ".simepr" / "user_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_user_model(name: str, components: list[SpinComponent], directory: Path | None = None) -> Path:
    """Persist a model pack to disk and return the file path."""
    directory = directory or default_user_dir()
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "user_model"
    path = directory / f"{safe}.simepr.json"
    path.write_text(components_to_json(components, name=name), encoding="utf-8")
    return path


def list_user_models(directory: Path | None = None) -> dict[str, Path]:
    """Return {model_name: path} for all saved model packs in the directory."""
    directory = directory or default_user_dir()
    out: dict[str, Path] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.simepr.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            name = meta.get("name", path.stem)
        except Exception:  # noqa: BLE001
            name = path.stem
        out[name] = path
    return out


def load_user_model(path: Path) -> tuple[list[SpinComponent], dict]:
    """Load a model pack from a file path."""
    return components_from_json(Path(path).read_text(encoding="utf-8"))
