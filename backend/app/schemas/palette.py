from enum import Enum

from pydantic import BaseModel, Field, field_validator


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
        default=None,
        description=(
            "Armonía a aplicar. Omitido o `null` = \"cualquier estilo\": se "
            "devuelven propuestas con tipos variados (rotación determinista de "
            "las 4 armonías). Un valor explícito filtra/rerankea el ranking a "
            "esa armonía."
        ),
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

    @field_validator("fallback", mode="before")
    @classmethod
    def _none_fallback_is_false(cls, value: bool | None) -> bool | None:
        """Acepta `fallback: null` (p. ej. al reenviar una paleta en /feedback)."""
        return False if value is None else value


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
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confianza de la armonía identificada. Si `confidenceCalibrated` es "
            "true, es la probabilidad calibrada (temperatura ajustada en validación)."
        ),
    )
    colors: list[Color]
    fallback: bool = False
    confidenceCalibrated: bool = Field(
        default=False,
        description="True si `confidence` viene de la calibración activa del manifiesto.",
    )
    needsRetry: bool = Field(
        default=False,
        description=(
            "True si la confianza es menor que `IDENTIFY_MIN_CONFIDENCE` y conviene "
            "probar con otra foto. Siempre false en el fallback predominante."
        ),
    )
    suggestion: str | None = Field(
        default=None,
        description="Sugerencia accionable en español cuando `needsRetry` es true.",
    )


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
