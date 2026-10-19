from fastapi import APIRouter

from app.config import APICOnfig
from app.schemas.palette import ColorRole, HarmonyType

harmony_router = APIRouter(
    prefix=f"{APICOnfig.prefix}/harmony", tags=["harmony", "palette"]
)
colors_router = APIRouter(
    prefix=f"{APICOnfig.prefix}/colors", tags=["colors", "palette"]
)


@harmony_router.get(
    "/",
    description="Lista de tipos de armonías para colores",
    response_description="Lista de tipos de armonías para colores",
)
async def harmony() -> list[HarmonyType]:
    return list(HarmonyType.__members__.values())


@colors_router.get(
    "/roles",
    description="Roles de color que el sistema puede asignar (primary, predominant, etc.).",
    response_description="Lista con los identificadores de los roles.",
)
async def get_color_roles() -> list[ColorRole]:
    return list(ColorRole.__members__.values())
