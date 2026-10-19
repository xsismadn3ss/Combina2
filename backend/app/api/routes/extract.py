from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import APICOnfig
from app.schemas.palette import ExtractResponse
from app.services.inference import extract_image_colors

router = APIRouter(prefix=f"{APICOnfig.prefix}/palette", tags=["palette"])


@router.post(
    "/extract",
    description=(
        "Extrae los 10 colores más frecuentes de una imagen con la red E "
        "(fallback clásico KMeans mientras no haya modelo activo). "
        "Devuelve colores ordenados por ponderancia con su peso."
    ),
    response_description="Top de colores predominantes con ponderancia.",
)
async def extract(file: Annotated[UploadFile, File(...)]) -> ExtractResponse:
    try:
        content = await file.read()
        return extract_image_colors(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
