from enum import Enum

from pydantic import BaseModel, Field


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
    weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ponderancia (fracción) del color en la imagen. Solo en extract/identify.",
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
        description="Armonía a aplicar (opcional). En Fase 4+ actúa como filtro del ranking.",
    )
    topK: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Cantidad de paletas candidatas solicitadas (ranking real desde Fase 4).",
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
    fallback: bool = Field(
        default=False,
        description="True si respondió el motor clásico porque no hay red activa.",
    )


class ScoredPalette(BaseModel):
    type: str
    colors: list[Color]
    score: float = Field(..., ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    palettes: list[ScoredPalette] = Field(
        ..., description="Paletas candidatas ordenadas por score descendente."
    )
    fallback: bool = False


class ExtractResponse(BaseModel):
    colors: list[Color] = Field(
        ..., description="Top-10 (o menos) colores predominantes con ponderancia."
    )
    fallback: bool = False


class IdentifyResponse(BaseModel):
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    colors: list[Color]
    fallback: bool = False


class FeedbackRequest(BaseModel):
    palette: PaletteResponse = Field(
        ..., description="Paleta sobre la que el usuario opina."
    )
    accepted: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    imageSha256: str | None = Field(
        default=None, description="SHA-256 de la imagen origen, si la hubo."
    )


class FeedbackResponse(BaseModel):
    ok: bool = True
