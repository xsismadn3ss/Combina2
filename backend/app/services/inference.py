"""Inferencia: redes TF si hay `.keras` activo, motor clásico si no.

Registro: `backend/models/manifest.json` -> {"<slot>": {"active": "<file>"|null}}.
TensorFlow se importa de forma perezosa: sin TF instalado o sin modelo activo,
todo responde por el fallback clásico con `fallback: true` (la API nunca se cae
por un modelo ausente).

El path con modelo E activo NO importa sklearn: `app.services.classic` solo se
importa dentro de las funciones de fallback.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.color_spaces import ColorConverter
from app.schemas.palette import (
    Color,
    ColorRole,
    ExtractResponse,
    GenerateResponse,
    HarmonyType,
    IdentifyResponse,
    ScoredPalette,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

_identifier: Any | None = None
_generator: Any | None = None
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


def _classic_identifier() -> Any:
    """Singleton perezoso: importar aquí para no traer sklearn en el path con red."""
    global _identifier
    if _identifier is None:
        from app.services.classic import ClassicPaletteIdentifier

        _identifier = ClassicPaletteIdentifier()
    return _identifier


def _classic_generator() -> Any:
    global _generator
    if _generator is None:
        from app.services.classic import ClassicPaletteGenerator

        _generator = ClassicPaletteGenerator()
    return _generator


def load_keras_model(slot: str) -> Any | None:
    """Carga perezosa del .keras activo del slot, o None si no hay red disponible."""
    if slot in _keras_cache:
        return _keras_cache[slot]
    path = active_keras_path(slot)
    if path is None:
        return None
    try:
        import sys

        import tensorflow as tf  # type: ignore[import-not-found]

        # TensorFlow/Keras importa sklearn vía wrappers aun sin usarlo;
        # lo purgamos para mantener el path con red E libre de sklearn
        # (ver bloqueo ALTA C3 del #19).
        for mod in list(sys.modules.keys()):
            if mod == "sklearn" or mod.startswith("sklearn."):
                del sys.modules[mod]
    except ImportError:
        return None
    custom_objects = None
    if slot == "extract":
        try:
            from app.ml.extract_model import EXTRACT_CUSTOM_OBJECTS

            custom_objects = EXTRACT_CUSTOM_OBJECTS
        except ImportError:
            custom_objects = None
    try:
        model = tf.keras.models.load_model(str(path), custom_objects=custom_objects)
        # load_model puede re-importar sklearn vía wrappers; purga de nuevo
        for mod in list(sys.modules.keys()):
            if mod == "sklearn" or mod.startswith("sklearn."):
                del sys.modules[mod]
    except Exception:
        return None
    _keras_cache[slot] = model
    return model


def extract_colors_from_outputs(centroids: Any, weights: Any) -> list[Color]:
    """Salidas de la red E ((10,3) Lab normalizado + (10,) pesos) → `Color`.

    Sin TF ni sklearn: filtra pesos < 0.01, renormaliza (suma 1.0) y ordena
    por peso. Lanza `ValueError` si las salidas son inválidas (el llamador
    cae al fallback clásico).
    """
    c = np.asarray(centroids, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if c.ndim == 3 and c.shape[0] == 1:
        c = c[0]
    if w.ndim == 2 and w.shape[0] == 1:
        w = w[0]
    if c.ndim != 2 or c.shape[1] != 3 or w.ndim != 1 or c.shape[0] != w.shape[0]:
        raise ValueError("Salidas de la red E con forma inválida")
    if c.shape[0] == 0:
        raise ValueError("La red E no devolvió colores")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Pesos de la red E inválidos")
    w = w / total
    order = np.argsort(-w, kind="stable")
    keep = [int(i) for i in order if float(w[i]) >= 0.01][:10]
    if not keep:
        keep = [int(order[0])]
    sel = w[keep]
    sel = sel / float(sel.sum())
    colors: list[Color] = []
    for idx, wi in zip(keep, sel):
        lab = ColorConverter.lab_denormalize(
            (float(c[idx, 0]), float(c[idx, 1]), float(c[idx, 2]))
        )
        lab = (
            float(np.clip(lab[0], 0.0, 100.0)),
            float(np.clip(lab[1], -128.0, 128.0)),
            float(np.clip(lab[2], -128.0, 128.0)),
        )
        colors.append(
            Color(
                value=ColorConverter.lab_to_hex(lab).upper(),
                role=ColorRole.PREDOMINANT.value,
                weight=float(wi),
            )
        )
    return colors


def extract_image_colors(image_bytes: bytes) -> ExtractResponse:
    model = load_keras_model("extract")
    if model is not None:
        colors = _extract_with_model(model, image_bytes)
        if colors is not None:
            return ExtractResponse(colors=colors, fallback=False)
    # Fallback clásico (único camino sin red E activa o si la red falla).
    return ExtractResponse(
        colors=_classic_identifier().identify_palette(image_bytes), fallback=True
    )


def _extract_with_model(model: Any, image_bytes: bytes) -> list[Color] | None:
    """Inferencia de la red E (Fase 3).

    Preproceso idéntico al común (`datasets.preprocess_image`: lado mayor
    300, 3000 px, Lab normalizado) → predict → parseo ordenado por peso.
    Retorna None ante cualquier fallo para caer al fallback clásico.
    """
    try:
        from app.ml.datasets import preprocess_image

        pre = preprocess_image(image_bytes)
        x = np.asarray(pre["pixels_lab"], dtype=np.float32).reshape(
            1, -1, 3
        )
        pred = model.predict(x, verbose=0)
        if isinstance(pred, (list, tuple)):
            centroids, weights = pred[0], pred[1]
        else:
            raise ValueError("Salidas de la red E con forma inválida")
        return extract_colors_from_outputs(centroids, weights)
    except Exception:
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
    classic = _classic_generator().generate_palette(base_hex_colors, harmony)
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
