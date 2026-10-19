from fastapi import APIRouter, HTTPException

from app.config import APICOnfig
from app.schemas.palette import CreatePaletteRequest, GenerateResponse, HarmonyType
from app.services.inference import generate_palettes

router = APIRouter(prefix=f"{APICOnfig.prefix}/palette", tags=["palette"])


@router.post(
    "/generate",
    description=(
        "Genera paletas candidatas a partir de 1+ colores base. "
        "La red G (Fase 4) devolverá un ranking con score; hoy responde el motor "
        "clásico HSL para la armonía pedida con `fallback: true`."
    ),
    response_description="Paletas candidatas ordenadas por score descendente.",
)
def generate(payload: CreatePaletteRequest) -> GenerateResponse:
    try:
        return generate_palettes(
            base_hex_colors=payload.colors,
            harmony=payload.harmony or HarmonyType.COMPLEMENTARY,
            top_k=payload.topK,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
