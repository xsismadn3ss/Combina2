from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

#1. DTOs (Data Transfer Objects) Pydantic Schemas
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
    value: str = Field(..., example="#FF5733", description="Código hexadecimal del color")
    role: str = Field(..., example="primary", description="Rol del color en la paleta")

class HarmonyType(str, Enum):
    COMPLEMENTARY = "complementario"
    TRIADIC = "triada"
    ANALOGOUS = "analogo"
    MONOCHROMATIC = "monocromatica"

class CreatePaletteRequest(BaseModel):
    colors: List[str] = Field(..., min_items=1, example=["#FF5733"], description="Colores base en formato hexadecimal")
    harmony: Optional[HarmonyType] = Field(
        default=HarmonyType.COMPLEMENTARY,
        description="Tipo de armonia matemática a aplicar (opcional)"
    )

class PaletteResponse(BaseModel):
    type: str = Field(..., example="triada", description="Tipo de paleta generada o identificada")
    colors: List[Color] = Field(..., description="Listado estructurado de objetos Color")