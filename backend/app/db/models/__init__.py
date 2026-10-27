"""Modelos ORM.

``feedback`` existe en las dos bases (runtime y entrenamiento). El resto de
tablas solo se crea/usa en la DB de entrenamiento; el mismo conjunto de
migraciones se aplica a ambas, así que el esquema es idéntico.

``feedback`` NO tiene FK a ``palettes``: es un snapshot de la paleta que vio el
usuario y puede referirse a una paleta que no esté en la DB de entrenamiento
(p. ej. una generada en producción). Por eso ``palette_json`` guarda el
snapshot completo y ``palette_type`` queda denormalizado.
"""

from app.db import base as _base  # noqa: F401,E402  (fija naming_convention)
from app.db.models.runtime import Feedback
from app.db.models.training import (
    ExtractedColor,
    Image,
    Palette,
    PaletteColor,
    TrainingRun,
)

__all__ = [
    "ExtractedColor",
    "Feedback",
    "Image",
    "Palette",
    "PaletteColor",
    "TrainingRun",
]
