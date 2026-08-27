from fastapi import APIRouter

from src.config import APICOnfig
from src.schemas.palette import ColorRole

router = APIRouter(prefix=f"{APICOnfig.prefix}/colors", tags=["colors", "pallete"])


@router.get(
    "/roles",
    description="Devuelve una lista dinámica con los roles de color que el sistema puede asignar (primary, secondary, accent, etc.).",
    response_description="Lista de cadenas de texto con los identificadores de los roles.",
)
async def get_color_roles() -> list[ColorRole]:
    enums = list(ColorRole.__members__.values())
    return enums
