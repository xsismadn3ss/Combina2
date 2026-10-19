from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import APICOnfig
from app.schemas.palette import IdentifyResponse
from app.services.inference import identify_image

router = APIRouter(prefix=f"{APICOnfig.prefix}/palette", tags=["palette"])


@router.post(
    "/identify",
    description=(
        "Identifica la paleta de una foto: extrae colores (red E) y clasifica la "
        "armonía que mejor la describe (red C). Sin modelos activos responde "
        "predominancia clásica con `fallback: true`."
    ),
    response_description="Tipo de paleta con confianza y colores extraídos.",
)
async def identify(file: Annotated[UploadFile, File(...)]) -> IdentifyResponse:
    try:
        content = await file.read()
        return identify_image(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
