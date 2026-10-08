from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.config import APICOnfig
from src.routes.examples.palette import (
    create_palette_examples,  # pyright: ignore[reportAny]
)
from src.schemas.palette import CreatePaletteRequest, HarmonyType, PaletteResponse
from src.services.palette_generator import ExpertSystemPaletteGenerator
from src.services.palette_identifier import ExpertSystemPaletteIdentifier

router = APIRouter(prefix=f"{APICOnfig.prefix}/palette", tags=["palette"])

se_generator = ExpertSystemPaletteGenerator()
se_identifier = ExpertSystemPaletteIdentifier()


@router.post(
    "/generate",
    description=(
        "Este endpoint aplica reglas geométricas fundamentadas en el círculo cromático "
        "sobre el espacio de color HSL para construir una paleta armónica.\n\n"
        "**Reglas geométricas soportadas:**\n"
        "* **Complementario:** Desplazamiento de +180 grados en el matiz.\n"
        "* **Triádico:** Desplazamientos de +120 y +240 grados.\n"
        "* **Análogo:** Desplazamiento de ±30 grados.\n"
        "* **Monocromático:** Modificación de la luminosidad (Lightness)."
    ),
    response_description="DTO estructurado con los colores resultantes y roles asignados.",
)
def generate(
    payload: CreatePaletteRequest = create_palette_examples,
) -> PaletteResponse:
    try:
        palette = se_generator.generate_palette(
            base_hex_colors=payload.colors,
            harmony=payload.harmony or HarmonyType.TRIADIC,
        )
        return palette
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/identify",
    description=(
        "Recibe una imagen y extrae sus colores predominantes mediante "
        "K-Means (aprendizaje no supervisado). Devuelve una paleta ordenada "
        "de mayor a menor predominancia."
    ),
    response_description="DTO estructurado con los colores predominantes identificados.",
)
async def identify(file: Annotated[UploadFile, File(...)]) -> PaletteResponse:
    """Identificar una paleta de colores predominantes a partir de una imagen."""
    try:
        content = await file.read()
        palette = se_identifier.identify_palette(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return palette
