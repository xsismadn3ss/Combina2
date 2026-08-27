from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# 1. DTOs (Data Transfer Objects) Pydantic Schemas
class ColorRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ACCENT = "accent"
    COMPLEMENTARY = "complementary"
    ANALOGOUS_1 = "analogous_1"
    ANALOGOUS_2 = "analogous_2"
    TRIADIC_1 = "triadic_1"
    TRIADIC_2 = "triadic_2"
    PREDOMINANT = "predominant"


class Color(BaseModel):
    value: str = Field(
        ...,
        json_schema_extra={"example": "#FF5733"},
        description="Código hexadecimal del color",
    )
    role: str = Field(
        ...,
        json_schema_extra={"example": "primary"},
        description="Rol del color en la paleta",
    )


class HarmonyType(str, Enum):
    COMPLEMENTARY = "complementario"
    TRIADIC = "triada"
    ANALOGOUS = "analogo"
    MONOCHROMATIC = "monocromatica"


class CreatePaletteRequest(BaseModel):
    colors: list[str] = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": ["#FF5733"]},
        description="Colores base en formato hexadecimal",
    )
    harmony: HarmonyType | None = Field(
        default=HarmonyType.COMPLEMENTARY,
        description="Tipo de armonia matemática a aplicar (opcional)",
    )


class PaletteResponse(BaseModel):
    type: str = Field(
        ...,
        json_schema_extra={"example": "triada"},
        description="Tipo de paleta generada o identificada",
    )
    colors: list[Color] = Field(
        ..., description="Listado estructurado de objetos Color"
    )
