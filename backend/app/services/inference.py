"""Inferencia: redes TF si hay `.keras` activo, motor clásico si no.

Registro: `backend/models/manifest.json` (ver `app.services.model_registry`),
con `active`/`previous` y sha256 por versión. La resolución de versión y la
caché de Keras viven en el registro, de forma que un rollback en caliente no
devuelve el modelo anterior (clave de caché `slot:archivo`). En producción se
recomienda reiniciar el proceso o llamar a `clear_model_cache()` tras promover
o hacer rollback, para liberar el modelo anterior de memoria.

TensorFlow se importa de forma perezosa: sin TF instalado o sin modelo activo,
todo responde por el fallback clásico con `fallback: true` (la API nunca se cae
por un modelo ausente).

El path con modelo E activo NO importa sklearn: `app.services.classic` solo se
importa dentro de las funciones de fallback.
"""

from pathlib import Path
from typing import Any

import numpy as np

from app.config import identify_min_confidence
from app.ml.color_features import (
    LOW_INFO_CHROMA_MAX,
    palette_features,
    palette_features_legacy,
    palette_is_low_information,
)
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
from app.services import model_registry

# Alias de compatibilidad (tests y scripts leen estas rutas).
MODELS_DIR = model_registry.MODELS_DIR
MANIFEST_PATH = model_registry.MANIFEST_PATH

_identifier: Any | None = None
_generator: Any | None = None
_keras_cache: dict[str, Any] = {}


def load_manifest() -> dict[str, Any]:
    """Manifiesto del registro (tolerante a ausencia/corrupción)."""
    return model_registry.load_manifest()


def active_keras_path(slot: str) -> Path | None:
    """Ruta resuelta del slot: activo verificado o, si falla, `previous`."""
    return model_registry.resolve(slot, manifest=load_manifest())


def load_calibration(slot: str) -> dict[str, Any]:
    """Metadatos de calibración de la versión **realmente resuelta**.

    Devuelve ``{}`` si falta el manifiesto, el slot/archivo o la clave
    ``calibration``. No importa TF ni sklearn: seguro en el path de inferencia.
    """
    try:
        manifest = load_manifest()
        path = active_keras_path(slot)
        if path is None:
            return {}
        entry = manifest.get(slot) or {}
        metadata = (entry.get("files") or {}).get(path.name) or {}
        calibration = metadata.get("calibration") if isinstance(metadata, dict) else None
        return calibration if isinstance(calibration, dict) else {}
    except Exception:
        return {}


def clear_model_cache() -> None:
    """Vacía la caché de modelos Keras (útil tras promover/rollback en caliente)."""
    _keras_cache.clear()


READINESS_SLOTS = ("extract", "generate", "classify")


def readiness_report() -> dict[str, Any]:
    """Estado real de los tres slots para el healthcheck de Railway.

    Para cada slot resuelve la versión activa con el registro (existencia +
    sha256) e intenta cargar el `.keras` mediante ``load_keras_model`` (usa la
    caché: el primer ``GET /ready`` puede tardar; los siguientes son baratos).

    Nunca lanza: cualquier fallo se reporta como ``loaded: false`` con un
    ``error`` breve. ``ready`` solo es ``True`` si los 3 slots activos cargan.
    No toca ``training.db``: en runtime solo se usan ``.keras`` + manifiesto.
    """
    models: dict[str, dict[str, Any]] = {}
    for slot in READINESS_SLOTS:
        entry: dict[str, Any] = {
            "active": None,
            "loaded": False,
            "fallback": True,
        }
        try:
            declared = model_registry.active_version(slot)
            entry["active"] = declared
            path = model_registry.resolve(slot)
            if path is None:
                entry["error"] = (
                    "sin versión activa verificable (archivo ausente o sha256 no coincide)"
                )
            else:
                # Si el activo cayó y se resolvió `previous`, se reporta el
                # archivo que realmente se usa.
                entry["active"] = path.name
                if load_keras_model(slot) is not None:
                    entry["loaded"] = True
                    entry["fallback"] = False
                else:
                    entry["error"] = (
                        "no se pudo cargar el modelo (TensorFlow ausente o carga fallida)"
                    )
        except Exception as exc:  # noqa: BLE001 - readiness nunca debe romper
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        models[slot] = entry

    ready = all(item["loaded"] for item in models.values())
    return {
        "ready": ready,
        "models": models,
        "fallback": not ready,
        "storage": "models_only",
    }


def _active_temperature(slot: str) -> float | None:
    """Temperatura válida de calibración activa, o None si no se aplica."""
    calibration = load_calibration(slot)
    if calibration.get("method") != "temperature":
        return None
    if calibration.get("calibrated") is not True:
        return None
    try:
        temperature = float(calibration.get("temperature"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(temperature) or temperature <= 0.0 or abs(temperature - 1.0) < 1e-12:
        return None
    return temperature


def _apply_active_platt(score: float) -> float:
    """Aplica Platt si G tiene ``calibration.calibrated: true``; si no, no-op."""
    calibration = load_calibration("generate")
    if calibration.get("method") != "platt" or calibration.get("calibrated") is not True:
        return score
    try:
        a = float(calibration.get("a"))
        b = float(calibration.get("b"))
    except (TypeError, ValueError):
        return score
    if not (np.isfinite(a) and np.isfinite(b)):
        return score
    from app.ml.calibration import apply_platt

    return float(np.clip(apply_platt(score, a, b), 0.0, 1.0))


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
    """Carga perezosa del `.keras` resuelto del slot, o None si no hay red.

    La caché se indexa por ``slot:archivo``: si se hace rollback o se activa
    otra versión, la siguiente llamada carga el archivo nuevo. Para liberar el
    modelo anterior en un proceso de larga vida, reinicia o usa
    ``clear_model_cache()``.
    """
    path = active_keras_path(slot)
    if path is None:
        return None
    cache_key = f"{slot}:{path.name}"
    if cache_key in _keras_cache:
        return _keras_cache[cache_key]
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
    # Descarta entradas antiguas del mismo slot para no retener modelos obsoletos.
    for key in [k for k in _keras_cache if k.startswith(f"{slot}:")]:
        _keras_cache.pop(key, None)
    _keras_cache[cache_key] = model
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
_RETRY_SUGGESTION = (
    "La foto no tiene suficiente variedad de color; prueba con mejor luz o una "
    "imagen más colorida."
)
_GRAY_SUGGESTION = (
    "La imagen es casi monocroma (gris o un solo tono); prueba con una foto "
    "con varios colores bien iluminada."
)
# Por debajo de este croma máximo la imagen puede no aportar señal de armonía;
# el guard exige además poca variedad de tono (ver `palette_is_low_information`).
COLORFULNESS_RETRY_MIN = LOW_INFO_CHROMA_MAX


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
        # Calibración Platt (no-op mientras generate-v1 esté rechazado por AUC).
        base_score = _apply_active_platt(base_score)
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
            calibrated = _active_temperature("classify") is not None
            low_information = palette_is_low_information(colors)
            low_confidence = confidence < identify_min_confidence()
            needs_retry = low_information or low_confidence
            if low_information:
                suggestion = _GRAY_SUGGESTION
            elif low_confidence:
                suggestion = _RETRY_SUGGESTION
            else:
                suggestion = None
            return IdentifyResponse(
                type=label,
                confidence=confidence,
                colors=colors,
                fallback=False,
                confidenceCalibrated=calibrated,
                needsRetry=needs_retry,
                suggestion=suggestion,
            )
    # Fallback: sin red C no hay clasificación; se reporta predominancia.
    # No tiene sentido pedir reintento: el fallback no intenta clasificar.
    return IdentifyResponse(
        type="predominant",
        confidence=1.0,
        colors=colors,
        fallback=True,
        confidenceCalibrated=False,
        needsRetry=False,
        suggestion=None,
    )


def _classify_input_dim(model: Any) -> int | None:
    """Dimensión de entrada del modelo C, o None si no se puede detectar.

    Soporta `model.input_shape` (tupla `(d,)` o `(None, d)` en Keras 3) y
    `model.inputs[0].shape[-1]`. Cualquier fake sin esos atributos → None
    (la inferencia usa entonces las features legacy de 40 dims).
    """
    shape = getattr(model, "input_shape", None)
    if shape is None:
        inputs = getattr(model, "inputs", None)
        try:
            if inputs:
                shape = inputs[0].shape
        except Exception:
            shape = None
    dim: Any = None
    if isinstance(shape, (list, tuple)):
        if shape and isinstance(shape[0], (list, tuple)):
            # lista de shapes de entrada (modelo multi-input): usa la primera
            dim = shape[0][-1] if shape[0] else None
        elif shape:
            dim = shape[-1]
    elif isinstance(shape, int):
        dim = shape
    try:
        dim = int(dim) if dim is not None else None
    except (TypeError, ValueError):
        return None
    return dim if dim is not None and dim > 0 else None


def _classify_with_model(
    model: Any, colors: list[Color]
) -> tuple[str, float] | None:
    """Inferencia de la red C sobre la paleta extraída.

    Detecta la dimensión de entrada del modelo: 54 dims (`classify-v2`,
    features de conjunto ordenadas por tono) o, por defecto/rollback,
    40 dims legacy (`classify-v1`: 10*Lab +10 pesos).

    Si el `.keras` activo tiene ``calibration.calibrated: true`` con
    ``method: temperature``, la confianza devuelta es la probabilidad
    calibrada; el ``type`` (argmax) es invariante a la temperatura.
    Retorna (type, confidence) o None si falla.
    """
    try:
        if not colors:
            return None
        input_dim = _classify_input_dim(model)
        if input_dim == 54:
            feat = palette_features(colors, input_dim=54)
        else:
            # 40 legacy, también cuando la dimensión no se puede detectar.
            feat = palette_features_legacy(colors)
        feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
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
        temperature = _active_temperature("classify")
        if temperature is not None:
            from app.ml.calibration import apply_temperature

            pred = apply_temperature(pred, temperature)
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
