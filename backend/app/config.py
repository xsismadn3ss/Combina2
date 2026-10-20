import os
from dataclasses import dataclass

DEFAULT_IDENTIFY_MIN_CONFIDENCE = 0.70


@dataclass(frozen=True)
class APICOnfig:
    prefix: str = "/api/v1"


def identify_min_confidence() -> float:
    """Umbral mínimo de confianza calibrada para ``/identify``.

    Configurable por ``IDENTIFY_MIN_CONFIDENCE`` (default 0.70). Valores fuera
    de ``[0, 1]`` o no numéricos se ignoran (se usa el default). El umbral es
    de producto, no de evaluación: sirve para pedir otra foto cuando la
    clasificación es dudosa. Justificación risk-coverage en ``docs/api.md`` y
    ``backend/eval/thresholds.json``.
    """
    raw = os.environ.get("IDENTIFY_MIN_CONFIDENCE")
    if raw is None:
        return DEFAULT_IDENTIFY_MIN_CONFIDENCE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_IDENTIFY_MIN_CONFIDENCE
    if not 0.0 <= value <= 1.0:
        return DEFAULT_IDENTIFY_MIN_CONFIDENCE
    return value
