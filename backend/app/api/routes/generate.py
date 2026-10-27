from fastapi import APIRouter, HTTPException

from app.config import APICOnfig
from app.schemas.palette import CreatePaletteRequest, GenerateResponse
from app.services.inference import generate_palettes

router = APIRouter(prefix=f"{APICOnfig.prefix}/palette", tags=["palette"])


@router.post(
    "/generate",
    description=(
        "Genera paletas candidatas a partir de 1+ colores base. Cada paleta "
        "conserva las anclas y las completa con acentos hasta 7 colores. Sin "
        "`harmony` (o con `null`) devuelve propuestas de tipos variados "
        "siguiendo una rotación determinista de las 4 armonías; con `harmony` "
        "explícita filtra/rerankea a ese tipo. La red G aporta el score; sin "
        "modelo responde el motor clásico HSL con `fallback: true`."
    ),
    response_description="Paletas candidatas ordenadas por score descendente.",
)
def generate(payload: CreatePaletteRequest) -> GenerateResponse:
    try:
        return generate_palettes(
            base_hex_colors=payload.colors,
            harmony=payload.harmony,
            top_k=payload.topK,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
