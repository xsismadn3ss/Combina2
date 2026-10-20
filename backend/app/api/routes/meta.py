from fastapi import APIRouter

from app.config import APICOnfig
from app.schemas.palette import ColorRole, HarmonyType
from app.services import model_registry

harmony_router = APIRouter(
    prefix=f"{APICOnfig.prefix}/harmony", tags=["harmony", "palette"]
)
colors_router = APIRouter(
    prefix=f"{APICOnfig.prefix}/colors", tags=["colors", "palette"]
)
models_router = APIRouter(prefix=APICOnfig.prefix, tags=["models"])


@harmony_router.get(
    "",
    description="Lista de tipos de armonías para colores",
    response_description="Lista de tipos de armonías para colores",
)
@harmony_router.get(
    "/",
    include_in_schema=False,
    description="Alias con barra final (compatibilidad)",
    response_description="Lista de tipos de armonías para colores",
)
async def harmony() -> list[HarmonyType]:
    # Ambas rutas existen para evitar el 307 con Location absoluto que rompía
    # el proxy same-origin del frontend (CORS).
    return list(HarmonyType.__members__.values())


@colors_router.get(
    "/roles",
    description="Roles de color que el sistema puede asignar (primary, predominant, etc.).",
    response_description="Lista con los identificadores de los roles.",
)
async def get_color_roles() -> list[ColorRole]:
    return list(ColorRole.__members__.values())


def _public_file_metadata(metadata: dict | None) -> dict:
    """Metadatos públicos de una versión (nunca rutas absolutas)."""
    data = metadata if isinstance(metadata, dict) else {}
    return {
        "sha256": data.get("sha256"),
        "createdAt": data.get("createdAt"),
        "gitSha": data.get("gitSha"),
        "trainingRunId": data.get("trainingRunId"),
        "tfVersion": data.get("tfVersion"),
        "kerasVersion": data.get("kerasVersion"),
        "metrics": data.get("metrics") or {},
        "calibration": data.get("calibration") or {},
    }


@models_router.get(
    "/models",
    description=(
        "Registro de versiones de modelos por slot (read-only). No carga "
        "TensorFlow ni los .keras; solo lee el manifiesto."
    ),
    response_description="Slot -> {active, previous, versions, files}",
)
async def get_models() -> dict[str, dict]:
    manifest = model_registry.load_manifest()
    result: dict[str, dict] = {}
    for slot in model_registry.list_slots():
        entry = manifest.get(slot) or {}
        if not isinstance(entry, dict):
            continue
        files = entry.get("files") or {}
        result[slot] = {
            "active": entry.get("active"),
            "previous": entry.get("previous"),
            "versions": list(entry.get("versions") or []),
            "files": {
                name: _public_file_metadata(metadata)
                for name, metadata in files.items()
                if isinstance(name, str)
            },
        }
    return result
