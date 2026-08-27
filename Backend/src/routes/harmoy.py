from fastapi.routing import APIRouter

from src.config import APICOnfig
from src.schemas.palette import HarmonyType

router = APIRouter(prefix=f"{APICOnfig.prefix}/harmony", tags=["harmony", "pallete"])


@router.get(
    "/",
    description="Lista de tipos de harmonías para colores",
    response_description="Lista de tipos de harmonías para colores",
)
async def harmony() -> list[HarmonyType]:
    enums = list(HarmonyType.__members__.values())
    return enums
