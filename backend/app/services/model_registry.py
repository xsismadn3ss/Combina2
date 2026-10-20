"""Registro de modelos `.keras`: versiones, hashes, activación y rollback (B3).

Módulo **puro Python** (sin TensorFlow, Keras ni sklearn). Es la única fuente de
verdad para leer/escribir `backend/models/manifest.json`; la inferencia y los
scripts de entrenamiento lo usan en lugar de manipular el JSON a mano.

Contrato del manifiesto (schemaVersion 2)::

    {
      "schemaVersion": 2,
      "updatedAt": "<ISO-8601 UTC>",
      "<slot>": {
        "active": "<archivo>.keras",
        "previous": "<archivo>.keras",      # opcional (solo rollback)
        "versions": ["<slot>-v1.keras", ...],
        "files": {
          "<archivo>.keras": {
            "sha256": "...", "createdAt": "...", "gitSha": "...",
            "trainingRunId": 17, "tfVersion": "...", "kerasVersion": "...",
            "metrics": {...}, "calibration": {...}
          }
        }
      }
    }

Política: **nunca se borra una versión**. Activar o hacer rollback solo mueve
`active`/`previous`; los archivos y sus metadatos se conservan.

Las constantes `MODELS_DIR` y `MANIFEST_PATH` se leen en cada llamada y pueden
monkeypatchearse en tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

SCHEMA_VERSION = 2
_META_KEYS = {"schemaVersion", "updatedAt"}


# --- Utilidades ---------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str | None:
    """sha256 hexadecimal del archivo, o ``None`` si no existe."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    """Versión instalada de un paquete sin importarlo (TF/Keras incluidos)."""
    try:
        return metadata.version(name)
    except Exception:
        return None


def framework_versions() -> tuple[str | None, str | None]:
    """``(tensorflow, keras)`` instalados, o ``None`` si no están."""
    return package_version("tensorflow"), package_version("keras")


def git_sha() -> str | None:
    """SHA corto del repo si está disponible; nunca falla."""
    try:
        from app.ml.benchmarks import git_sha as _git_sha

        return _git_sha()
    except Exception:
        return None


# --- Manifiesto ---------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    """Lee el manifiesto. Si falta o está corrupto devuelve uno vacío válido."""
    path = MANIFEST_PATH
    if not path.exists():
        return {"schemaVersion": SCHEMA_VERSION, "updatedAt": _now()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("manifest ilegible en %s; se usa uno vacío", path)
        return {"schemaVersion": SCHEMA_VERSION, "updatedAt": _now()}
    if not isinstance(data, dict):
        return {"schemaVersion": SCHEMA_VERSION, "updatedAt": _now()}
    return data


def save_manifest(manifest: dict[str, Any]) -> None:
    """Escribe el manifiesto de forma atómica (temp en el mismo dir + replace)."""
    manifest["schemaVersion"] = SCHEMA_VERSION
    manifest["updatedAt"] = _now()
    path = MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def list_slots() -> list[str]:
    """Slots presentes en el manifiesto (excluye metadatos de cabecera)."""
    manifest = load_manifest()
    return [key for key, value in manifest.items() if isinstance(value, dict)]


def active_version(slot: str) -> str | None:
    entry = load_manifest().get(slot) or {}
    active = entry.get("active") if isinstance(entry, dict) else None
    return active or None


def previous_version(slot: str) -> str | None:
    entry = load_manifest().get(slot) or {}
    previous = entry.get("previous") if isinstance(entry, dict) else None
    return previous or None


def _verified_path(
    slot: str, filename: str, files: dict[str, Any]
) -> tuple[Path | None, str]:
    """Devuelve ``(path, estado)`` con estado ok|missing|mismatch|unhashed."""
    path = Path(MODELS_DIR) / filename
    if not path.exists():
        return None, "missing"
    metadata_entry = files.get(filename) if isinstance(files, dict) else None
    expected = (
        metadata_entry.get("sha256") if isinstance(metadata_entry, dict) else None
    )
    if not expected:
        # Manifiesto legacy sin hash: no se puede verificar, pero se tolera.
        return path, "unhashed"
    if sha256_file(path) != expected:
        return None, "mismatch"
    return path, "ok"


def resolve(slot: str, *, manifest: dict[str, Any] | None = None) -> Path | None:
    """Ruta del `.keras` a usar para el slot.

    Orden: versión `active` si existe y su sha256 coincide; si falta o el hash
    no coincide, se intenta `previous` (registrando un warning); si tampoco,
    ``None``. Manifiestos legacy sin `files`/`sha256` se aceptan sin verificar.
    """
    data = manifest if manifest is not None else load_manifest()
    entry = data.get(slot) or {}
    if not isinstance(entry, dict):
        return None
    files = entry.get("files") or {}

    active = entry.get("active")
    if active:
        path, status = _verified_path(slot, active, files)
        if status in ("ok", "unhashed"):
            return path
        reason = "no existe" if status == "missing" else "sha256 no coincide"
        logger.warning(
            "modelo activo %s:%s no válido (%s); se intenta previous",
            slot,
            active,
            reason,
        )

    previous = entry.get("previous")
    if previous and previous != active:
        path, status = _verified_path(slot, previous, files)
        if status in ("ok", "unhashed"):
            logger.warning(
                "modelo %s resuelto desde previous=%s (%s)",
                slot,
                previous,
                "sin hash verificable" if status == "unhashed" else "hash OK",
            )
            return path
        if status in ("missing", "mismatch"):
            logger.warning("previous %s:%s tampoco es válido", slot, previous)
    return None


# --- Activación / rollback ----------------------------------------------------


def _entry(manifest: dict[str, Any], slot: str) -> dict[str, Any]:
    entry = manifest.get(slot)
    if not isinstance(entry, dict):
        entry = {"active": None, "versions": [], "files": {}}
        manifest[slot] = entry
    entry.setdefault("versions", [])
    entry.setdefault("files", {})
    return entry


def activate(slot: str, filename: str, note: str | None = None) -> dict[str, Any]:
    """Activa ``filename`` (debe existir en ``MODELS_DIR``) y actualiza su hash."""
    path = Path(MODELS_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"no existe {path}")
    manifest = load_manifest()
    entry = _entry(manifest, slot)
    sha = sha256_file(path)
    versions: list[str] = entry["versions"]
    if filename not in versions:
        versions.append(filename)
    metadata_entry = entry["files"].get(filename)
    if not isinstance(metadata_entry, dict):
        metadata_entry = {}
    metadata_entry["sha256"] = sha
    metadata_entry.setdefault("createdAt", _now())
    if note:
        metadata_entry["note"] = note
    entry["files"][filename] = metadata_entry

    previous = entry.get("active")
    if previous and previous != filename:
        entry["previous"] = previous
    entry["active"] = filename
    save_manifest(manifest)
    return {
        "slot": slot,
        "active": filename,
        "previous": entry.get("previous"),
        "sha256": sha,
    }


def rollback(slot: str) -> dict[str, str | None]:
    """Intercambia `active` y `previous` (rollback reversible, sin borrar nada)."""
    manifest = load_manifest()
    entry = manifest.get(slot) or {}
    previous = entry.get("previous") if isinstance(entry, dict) else None
    if not previous:
        raise ValueError(f"el slot {slot!r} no tiene 'previous' para rollback")
    prev_path = Path(MODELS_DIR) / previous
    if not prev_path.exists():
        raise FileNotFoundError(f"no existe la versión previous {prev_path}")
    current = entry.get("active")
    entry["active"] = previous
    entry["previous"] = current
    save_manifest(manifest)
    return {"from": current, "to": previous}


# --- Verificación -------------------------------------------------------------


def _files_to_check(entry: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for name in entry.get("versions") or []:
        if isinstance(name, str) and name not in names:
            names.append(name)
    for key in ("active", "previous"):
        name = entry.get(key)
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def verify(slot: str | None = None, *, raise_on_error: bool = False) -> dict[str, Any]:
    """Comprueba existencia y sha256 de todas las versiones listadas.

    Devuelve ``{ok, missing, hash_mismatch, unhashed}`` donde cada entrada es
    ``{"slot", "file", ...}``. Por defecto no lanza: el resultado es inspeccionable.
    """
    manifest = load_manifest()
    slots = [slot] if slot else list_slots()
    missing: list[dict[str, Any]] = []
    hash_mismatch: list[dict[str, Any]] = []
    unhashed: list[dict[str, Any]] = []

    for slot_name in slots:
        entry = manifest.get(slot_name) or {}
        if not isinstance(entry, dict):
            continue
        files = entry.get("files") or {}
        for filename in _files_to_check(entry):
            path = Path(MODELS_DIR) / filename
            if not path.exists():
                missing.append({"slot": slot_name, "file": filename})
                continue
            metadata_entry = files.get(filename) if isinstance(files, dict) else None
            expected = (
                metadata_entry.get("sha256")
                if isinstance(metadata_entry, dict)
                else None
            )
            if not expected:
                unhashed.append({"slot": slot_name, "file": filename})
                continue
            actual = sha256_file(path)
            if actual != expected:
                hash_mismatch.append(
                    {
                        "slot": slot_name,
                        "file": filename,
                        "expected": expected,
                        "actual": actual,
                    }
                )

    result = {
        "ok": not missing and not hash_mismatch,
        "missing": missing,
        "hash_mismatch": hash_mismatch,
        "unhashed": unhashed,
    }
    if raise_on_error and not result["ok"]:
        raise FileNotFoundError(f"verificación de modelos fallida: {result}")
    return result


# --- Registro de nuevas versiones ---------------------------------------------


def parse_version(value: int | str) -> int:
    """Normaliza ``3``, ``"3"`` o ``"v3"`` a ``3``."""
    if isinstance(value, int):
        version = value
    else:
        text = str(value).strip().lower()
        if text.startswith("v"):
            text = text[1:]
        if not re.fullmatch(r"\d+", text):
            raise ValueError(f"versión inválida: {value!r} (usa 3 o v3)")
        version = int(text)
    if version <= 0:
        raise ValueError(f"la versión debe ser >= 1: {value!r}")
    return version


def version_filename(slot: str, version: int | str) -> str:
    return f"{slot}-v{parse_version(version)}.keras"


def next_free_version(slot: str) -> int:
    """Siguiente ``vN`` libre mirando archivos y manifiesto."""
    highest = 0
    pattern = re.compile(rf"^{re.escape(slot)}-v(\d+)\.keras$")
    for path in Path(MODELS_DIR).glob(f"{slot}-v*.keras"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    manifest = load_manifest()
    entry = manifest.get(slot) or {}
    if isinstance(entry, dict):
        names = list(entry.get("versions") or [])
        names.extend([entry.get("active"), entry.get("previous")])
        for name in names:
            if isinstance(name, str):
                match = pattern.match(name)
                if match:
                    highest = max(highest, int(match.group(1)))
    return highest + 1


def target_path(
    slot: str,
    *,
    version: int | str | None = None,
    out: str | Path | None = None,
) -> Path:
    """Ruta destino para un nuevo entrenamiento.

    Prioridad: ``out`` explícito > ``version`` normalizada > siguiente ``vN``
    libre. No comprueba existencia (eso lo hace el llamador para fallar antes
    de entrenar).
    """
    if out:
        return Path(out)
    if version is not None:
        return Path(MODELS_DIR) / version_filename(slot, version)
    return Path(MODELS_DIR) / version_filename(slot, next_free_version(slot))


def register_version(
    slot: str,
    path: str | Path,
    *,
    metrics: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    training_run_id: int | None = None,
    note: str | None = None,
    force: bool = False,
    calibration: dict[str, Any] | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Copia/registra el `.keras` en ``MODELS_DIR`` y lo activa.

    - Si el destino ya existe y no ``force`` → ``FileExistsError`` (nunca
      se sobrescribe una versión por accidente).
    - Calcula sha256, detecta versiones TF/Keras instaladas y escribe metadatos.
    - Añade a ``versions``, mueve ``previous`` al activo anterior y activa.

    ``filename`` permite registrar un archivo temporal con el nombre final
    (los scripts guardan en ``.tmp-<nombre>`` y registran como ``<nombre>``).
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no existe el modelo a registrar: {source}")
    destination = Path(MODELS_DIR) / (filename or source.name)
    same_file = source.resolve() == destination.resolve()
    if destination.exists() and not same_file and not force:
        raise FileExistsError(
            f"{destination} ya existe; usa --force para sobrescribirlo"
        )

    if not same_file:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f".tmp-{destination.name}")
        shutil.copy2(source, tmp)
        os.replace(tmp, destination)

    sha = sha256_file(destination)
    tf_version, keras_version = framework_versions()
    manifest = load_manifest()
    entry = _entry(manifest, slot)
    versions: list[str] = entry["versions"]
    if destination.name not in versions:
        versions.append(destination.name)

    metadata_entry = entry["files"].get(destination.name)
    if not isinstance(metadata_entry, dict):
        metadata_entry = {}
    metadata_entry.update(
        {
            "sha256": sha,
            "createdAt": _now(),
            "gitSha": git_sha(),
            "trainingRunId": training_run_id,
            "tfVersion": tf_version,
            "kerasVersion": keras_version,
            "metrics": metrics or {},
        }
    )
    if params is not None:
        metadata_entry["params"] = params
    if calibration is not None:
        metadata_entry["calibration"] = calibration
    if note:
        metadata_entry["note"] = note
    entry["files"][destination.name] = metadata_entry

    previous = entry.get("active")
    if previous and previous != destination.name:
        entry["previous"] = previous
    entry["active"] = destination.name
    save_manifest(manifest)
    return {
        "slot": slot,
        "filename": destination.name,
        "keras_path": str(destination),
        "sha256": sha,
        "training_run_id": training_run_id,
        "active": destination.name,
        "previous": entry.get("previous"),
    }
