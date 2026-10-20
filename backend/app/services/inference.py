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
    elif slot == "generate":
        try:
            from app.ml.generate_model import GENERATE_CUSTOM_OBJECTS

            custom_objects = GENERATE_CUSTOM_OBJECTS
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


_HARMONY_LABELS = ["complementario", "triada", "analogo", "monocromatica"]
_HARMONY_TO_IDX = {h: i for i, h in enumerate(_HARMONY_LABELS)}


def generate_palettes(
    base_hex_colors: list[str], harmony: HarmonyType, top_k: int
) -> GenerateResponse:
    model = load_keras_model("generate")
    if model is not None:
        palettes = _generate_with_model(model, base_hex_colors, harmony, top_k)
        if palettes is not None:
            return GenerateResponse(
                palettes=_condition_palettes(base_hex_colors, harmony, top_k, palettes),
                fallback=False,
            )
    from app.ml.color_spaces import ColorConverter

    # Los colores elegidos son anclas, no candidatos a sustituirse. Se
    # conservan al inicio de cada propuesta y la armonía solo completa la paleta.
    anchors = [color.upper() for color in base_hex_colors]
    hue, saturation, lightness = ColorConverter.hex_to_hsl(anchors[0])
    harmony_offsets = {
        HarmonyType.ANALOGOUS: (-30.0, 30.0),
        HarmonyType.TRIADIC: (120.0, 240.0),
        HarmonyType.COMPLEMENTARY: (180.0, 0.0),
        HarmonyType.MONOCHROMATIC: (0.0, 0.0),
    }.get(harmony, (180.0, 0.0))
    variants: list[ScoredPalette] = []
    for index in range(max(1, int(top_k))):
        colors = [Color(value=value, role="primary" if i == 0 else "secondary")
                  for i, value in enumerate(anchors)]
        variation = float(index) * 22.0
        if harmony == HarmonyType.MONOCHROMATIC:
            tones = [
                ColorConverter.hsl_to_hex(hue, max(35.0, saturation), tone_lightness).upper()
                for tone_lightness in (max(12.0, lightness - 28.0 - variation),
                                       min(92.0, lightness + 28.0 + variation))
            ]
        else:
            offsets = (harmony_offsets[0] - variation, harmony_offsets[1] + variation)
            tones = [
                ColorConverter.hsl_to_hex((hue + offset) % 360.0, max(45.0, saturation),
                                          min(85.0, max(25.0, lightness + (variation if j else -variation))))
                .upper()
                for j, offset in enumerate(offsets)
            ]
        for tone in tones:
            if len(colors) >= max(5, len(anchors)):
                break
            if tone not in {color.value for color in colors}:
                colors.append(Color(value=tone, role="accent"))
        # Para entradas coincidentes, completa con tonos claramente separados.
        tone_step = 0
        while len(colors) < max(5, len(anchors)):
            tone_step += 1
            tone = ColorConverter.hsl_to_hex(
                (hue + variation + tone_step * 67.0) % 360.0,
                max(45.0, saturation),
                (25.0, 75.0, 40.0, 65.0)[tone_step % 4],
            ).upper()
            if tone not in {color.value for color in colors}:
                colors.append(Color(value=tone, role="accent"))
        variants.append(ScoredPalette(type=harmony.value, colors=colors,
                                      score=round(1.0 - index * 0.01, 2)))
    return GenerateResponse(palettes=variants, fallback=True)


def _condition_palettes(
    anchors: list[str], harmony: HarmonyType, top_k: int,
    model_palettes: list[ScoredPalette],
) -> list[ScoredPalette]:
    """Keep every selected color fixed; generate distinct, harmony-conditioned accents.

    The learned head is useful for its compatibility score, but its single output
    is not a sampling distribution. We therefore condition its output with
    deliberately separated harmonic constructions rather than adding Lab jitter.
    """
    if len(anchors) > 5:
        raise ValueError("Se pueden seleccionar como máximo cinco colores")
    hue, saturation, lightness = ColorConverter.hex_to_hsl(anchors[0])
    offsets = {
        HarmonyType.ANALOGOUS: (-30.0, 30.0),
        HarmonyType.TRIADIC: (120.0, 240.0),
        HarmonyType.COMPLEMENTARY: (180.0, 0.0),
        HarmonyType.MONOCHROMATIC: (0.0, 0.0),
    }.get(harmony, (180.0, 0.0))
    result = []
    for index, proposed in enumerate(model_palettes[:max(1, int(top_k))]):
        colors = [Color(value=h.upper(), role="primary" if i == 0 else "secondary")
                  for i, h in enumerate(anchors)]
        rotation = index * 37.0
        for j in range(5 - len(colors)):
            # The G head supplies chroma/lightness guidance for each generated
            # slot; hue is projected onto the requested harmony instead of
            # blindly accepting a prediction that may replace an anchor.
            proposal_color = proposed.colors[min(3 + j, len(proposed.colors) - 1)]
            _, proposal_s, proposal_l = ColorConverter.hex_to_hsl(proposal_color.value)
            if harmony == HarmonyType.MONOCHROMATIC:
                tone_l = (15.0 + (index * 19.0 + j * 47.0) % 75.0)
                tone_l = min(90.0, max(10.0, tone_l * 0.7 + proposal_l * 0.3))
                tone = ColorConverter.hsl_to_hex(
                    hue, max(35.0, saturation * 0.7 + proposal_s * 0.3), tone_l
                )
            else:
                offset = offsets[j % 2] + rotation
                tone_s = min(95.0, max(45.0,
                    saturation * 0.7 + proposal_s * 0.3 + (index % 3 - 1) * 8.0))
                tone_l = min(82.0, max(25.0,
                    lightness * 0.7 + proposal_l * 0.3 + (j * 22.0 - 10.0)
                    + (index % 3 - 1) * 9.0))
                tone = ColorConverter.hsl_to_hex((hue + offset) % 360.0, tone_s, tone_l)
            colors.append(Color(value=tone.upper(), role=ColorRole.ACCENT.value))
        result.append(ScoredPalette(type=harmony.value, colors=colors,
                                    score=proposed.score))
    return result


def _generate_with_model(
    model: Any, base_hex_colors: list[str], harmony: HarmonyType, top_k: int
) -> list[ScoredPalette] | None:
    """Inferencia de la red G.

    Prepro: HEX→Lab normalizado + máscara + embedding armonía.
    Post: Lab→HEX + score + ranking topK, honra harmony como filtro/reranking.
    Retorna None si falla para caer al fallback clásico.
    """
    try:
        if not base_hex_colors:
            raise ValueError("Se requiere al menos un color base")
        # valida HEX y construye entrada
        palette_lab = np.zeros((1, 10, 3), dtype=np.float32)
        mask = np.zeros((1, 10), dtype=np.float32)
        harmony_oh = np.zeros((1, 4), dtype=np.float32)
        # harmony enum -> one-hot
        h_val = harmony.value if isinstance(harmony, HarmonyType) else str(harmony)
        if h_val in _HARMONY_TO_IDX:
            harmony_oh[0, _HARMONY_TO_IDX[h_val]] = 1.0
        # base colors (máx 10)
        for i, hx in enumerate(base_hex_colors[:10]):
            # valida HEX
            try:
                lab_raw = ColorConverter.hex_to_lab(hx)
            except ValueError as e:
                raise ValueError(str(e))
            # hex_to_lab ya valida; normaliza
            nlab = ColorConverter.lab_normalize(lab_raw)
            palette_lab[0, i, 0] = float(nlab[0])
            palette_lab[0, i, 1] = float(nlab[1])
            palette_lab[0, i, 2] = float(nlab[2])
            mask[0, i] = 1.0
        # inferencia
        # modelo espera [palette_lab, mask, harmony] en ese orden
        # soporta tanto lista como dict
        try:
            pred = model.predict([palette_lab, mask, harmony_oh], verbose=0)
        except Exception:
            # intenta dict por si el modelo se guardó con nombres distintos
            pred = model.predict(
                {"palette_lab": palette_lab, "mask": mask, "harmony": harmony_oh},
                verbose=0,
            )
        if not isinstance(pred, (list, tuple)) or len(pred) != 2:
            raise ValueError("Salidas G inválidas")
        pal_flat, score_arr = pred
        pal_flat = np.asarray(pal_flat, dtype=np.float32)
        score_arr = np.asarray(score_arr, dtype=np.float32)
        if pal_flat.ndim == 2 and pal_flat.shape[0] == 1:
            pal_flat = pal_flat[0]
        else:
            pal_flat = pal_flat.reshape(-1)
        if score_arr.ndim == 2 and score_arr.shape[0] == 1:
            base_score = float(score_arr[0, 0])
        elif score_arr.ndim == 1:
            base_score = float(score_arr[0])
        else:
            base_score = float(np.asarray(score_arr).reshape(-1)[0])
        base_score = float(np.clip(base_score, 0.0, 1.0))
        # base palette 15 -> 5x3
        if pal_flat.size != 15:
            raise ValueError("Paleta G debe ser 15 valores")
        base_lab_norm = pal_flat.reshape(5, 3)
        # clipping
        base_lab_norm[:, 0] = np.clip(base_lab_norm[:, 0], 0.0, 1.0)
        base_lab_norm[:, 1] = np.clip(base_lab_norm[:, 1], -1.0, 1.0)
        base_lab_norm[:, 2] = np.clip(base_lab_norm[:, 2], -1.0, 1.0)
        # Asegura que los colores base se preserven exactos (copia directa)
        n_base = int(mask[0].sum())
        for bi in range(min(n_base, 5)):
            base_lab_norm[bi] = palette_lab[0, bi]

        # genera top_k variantes diversas (determinista por contenido base)
        roles = [
            ColorRole.PRIMARY.value,
            ColorRole.SECONDARY.value,
            ColorRole.ACCENT.value,
            ColorRole.COMPLEMENTARY.value,
            ColorRole.ANALOGOUS_1.value,
        ]
        pal_type = h_val if h_val in _HARMONY_LABELS else HarmonyType.COMPLEMENTARY.value
        variants: list[ScoredPalette] = []
        # seed determinista basada en contenido
        import hashlib

        seed_base = int(hashlib.sha256("|".join([c.upper() for c in base_hex_colors[:3]]).encode()).hexdigest()[:8], 16) % (2**31 - 1)
        for k in range(int(top_k)):
            if k == 0:
                lab_var = base_lab_norm.copy()
                s = base_score
            else:
                rng = np.random.RandomState((seed_base + k * 1009) % (2**31 - 1))
                noise = rng.normal(0.0, 0.015 * k, size=(5, 3)).astype(np.float32)
                lab_var = base_lab_norm + noise
                # preserva bases exactas también en variantes (no ruidizar)
                for bi in range(min(n_base, 5)):
                    lab_var[bi] = base_lab_norm[bi]
                lab_var[:, 0] = np.clip(lab_var[:, 0], 0.0, 1.0)
                lab_var[:, 1] = np.clip(lab_var[:, 1], -1.0, 1.0)
                lab_var[:, 2] = np.clip(lab_var[:, 2], -1.0, 1.0)
                # score decae ligeramente
                s = float(np.clip(base_score * (1 - 0.04 * k) + rng.normal(0, 0.008), 0.0, 1.0))
            colors: list[Color] = []
            for idx in range(5):
                nlab = (float(lab_var[idx, 0]), float(lab_var[idx, 1]), float(lab_var[idx, 2]))
                lab_raw = ColorConverter.lab_denormalize(nlab)
                lab_raw = (
                    float(np.clip(lab_raw[0], 0.0, 100.0)),
                    float(np.clip(lab_raw[1], -128.0, 128.0)),
                    float(np.clip(lab_raw[2], -128.0, 128.0)),
                )
                hexv = ColorConverter.lab_to_hex(lab_raw).upper()
                colors.append(Color(value=hexv, role=roles[idx]))
            variants.append(ScoredPalette(type=pal_type, colors=colors, score=float(s)))
        # ranking desc por score
        variants.sort(key=lambda p: p.score, reverse=True)
        return variants[: int(top_k)]
    except ValueError:
        raise
    except Exception:
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
    """Inferencia de la red C sobre la paleta extraída.

    Entrada 40 dims (10*Lab +10 pesos) desde Colors con weight.
    Retorna (type, confidence) o None si falla.
    """
    try:
        if not colors:
            return None
        labs: list[tuple[float, float, float]] = []
        weights: list[float] = []
        for c in colors[:10]:
            try:
                lab_raw = ColorConverter.hex_to_lab(c.value)
            except ValueError:
                return None
            nlab = ColorConverter.lab_normalize(lab_raw)
            labs.append(nlab)
            w = c.weight if c.weight is not None else 1.0 / max(len(colors), 1)
            weights.append(float(w))
        while len(labs) < 10:
            labs.append(labs[-1] if labs else (0.5, 0.0, 0.0))
            weights.append(0.0)
        # normaliza pesos si no suman 1 (por padding)
        w_arr = np.array(weights[:10], dtype=np.float64)
        w_arr = np.clip(w_arr, 0.0, None)
        total = float(w_arr.sum())
        if not np.isfinite(total) or total <= 0:
            return None
        w_arr = w_arr / total
        labs_arr = np.array(labs[:10], dtype=np.float32).reshape(-1)  # 30
        feat = np.concatenate([labs_arr, w_arr.astype(np.float32)])[None, :].astype(np.float32)  # (1,40)
        pred = model.predict(feat, verbose=0)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 2 and pred.shape[0] == 1:
            pred = pred[0]
        else:
            pred = pred.reshape(-1)
        if pred.size != 4:
            return None
        # softmax ya aplicado; si no, normaliza
        s = float(pred.sum())
        if abs(s - 1.0) > 1e-3 and s > 0:
            pred = pred / s
        idx = int(np.argmax(pred))
        if idx < 0 or idx >= len(_HARMONY_LABELS):
            return None
        label = _HARMONY_LABELS[idx]
        conf = float(np.clip(pred[idx], 0.0, 1.0))
        return label, conf
    except Exception:
        return None


def lab_palette_from_hex(hex_colors: list[str]) -> list[tuple[float, float, float]]:
    """Utilidad para fases 3-4: HEX -> Lab normalizado (entrada de G/C)."""
    return [
        ColorConverter.lab_normalize(ColorConverter.hex_to_lab(h)) for h in hex_colors
    ]
