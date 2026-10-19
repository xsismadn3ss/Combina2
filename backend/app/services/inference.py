"""Inferencia: redes TF si hay `.keras` activo, motor clásico si no.

Registro: `backend/models/manifest.json` -> {"<slot>": {"active": "<file>"|null}}.
TensorFlow se importa de forma perezosa: sin TF instalado o sin modelo activo,
todo responde por el fallback clásico con `fallback: true` (la API nunca se cae
por un modelo ausente).
"""

import json
from pathlib import Path
from typing import Any

from app.ml.color_spaces import ColorConverter
from app.schemas.palette import (
    Color,
    ExtractResponse,
    GenerateResponse,
    HarmonyType,
    IdentifyResponse,
    ScoredPalette,
)
from app.services.classic import ClassicPaletteGenerator, ClassicPaletteIdentifier

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

_generator = ClassicPaletteGenerator()
_identifier = ClassicPaletteIdentifier()
_keras_cache: dict[str, Any] = {}


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def active_keras_path(slot: str) -> Path | None:
    active = load_manifest().get(slot, {}).get("active")
    if not active:
        return None
    path = MODELS_DIR / active
    return path if path.exists() else None


def load_keras_model(slot: str) -> Any | None:
    """Carga perezosa del .keras activo del slot, o None si no hay red disponible."""
    if slot in _keras_cache:
        return _keras_cache[slot]
    path = active_keras_path(slot)
    if path is None:
        return None
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError:
        return None
    model = tf.keras.models.load_model(str(path))
    _keras_cache[slot] = model
    return model


def extract_image_colors(image_bytes: bytes) -> ExtractResponse:
    model = load_keras_model("extract")
    if model is not None:
        colors = _extract_with_model(model, image_bytes)
        if colors is not None:
            return ExtractResponse(colors=colors, fallback=False)
    # Fallback clásico (Fase 1: único camino real).
    return ExtractResponse(colors=_identifier.identify_palette(image_bytes), fallback=True)


def _extract_with_model(model: Any, image_bytes: bytes) -> list[Color] | None:
    """Inferencia de la red E. Se implementa en Fase 3; hoy retorna None (fallback)."""
    _ = (model, image_bytes)
    return None


def generate_palettes(
    base_hex_colors: list[str], harmony: HarmonyType, top_k: int
) -> GenerateResponse:
    _ = top_k  # el ranking multi-candidata llega en Fase 4 (red G)
    model = load_keras_model("generate")
    if model is not None:
        palettes = _generate_with_model(model, base_hex_colors, harmony, top_k)
        if palettes is not None:
            return GenerateResponse(palettes=palettes, fallback=False)
    classic = _generator.generate_palette(base_hex_colors, harmony)
    return GenerateResponse(
        palettes=[
            ScoredPalette(type=classic.type, colors=classic.colors, score=1.0)
        ],
        fallback=True,
    )


def _generate_with_model(
    model: Any, base_hex_colors: list[str], harmony: HarmonyType, top_k: int
) -> list[ScoredPalette] | None:
    """Inferencia de la red G. Se implementa en Fase 4; hoy retorna None (fallback)."""
    _ = (model, base_hex_colors, harmony, top_k)
    return None


def identify_image(image_bytes: bytes) -> IdentifyResponse:
    colors = extract_image_colors(image_bytes).colors
    model = load_keras_model("classify")
    if model is not None:
        result = _classify_with_model(model, colors)
        if result is not None:
            label, confidence = result
            return IdentifyResponse(
                type=label, confidence=confidence, colors=colors, fallback=False
            )
    # Fallback: sin red C no hay clasificación; se reporta predominancia.
    return IdentifyResponse(
        type="predominant", confidence=1.0, colors=colors, fallback=True
    )


def _classify_with_model(
    model: Any, colors: list[Color]
) -> tuple[str, float] | None:
    """Inferencia de la red C sobre la paleta extraída. Fase 4; hoy None (fallback)."""
    _ = (model, colors)
    return None


def lab_palette_from_hex(hex_colors: list[str]) -> list[tuple[float, float, float]]:
    """Utilidad para fases 3-4: HEX -> Lab normalizado (entrada de G/C)."""
    return [
        ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hex_colors
    ]
