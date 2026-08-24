from fastapi import APIRouter, HTTPException, Body
from src.schemas.palette import CreatePaletteRequest, PaletteResponse, HarmonyType
from src.services.palette_generator import ExpertSystemPaletteGenerator

router = APIRouter(prefix="/api/v1/palettes", tags=["Generador de Paletas"])
se_generator = ExpertSystemPaletteGenerator()

@router.post(
    "/generate-harmonic",
    response_model=PaletteResponse,
    summary="Generar paleta armónica (Sistema Experto 1)",
    description=(
        "Este endpoint aplica reglas geométricas fundamentadas en el círculo cromático "
        "sobre el espacio de color HSL para construir una paleta armónica.\n\n"
        "**Reglas geométricas soportadas:**\n"
        "* **Complementario:** Desplazamiento de +180 grados en el matiz.\n"
        "* **Triádico:** Desplazamientos de +120 y +240 grados.\n"
        "* **Análogo:** Desplazamiento de ±30 grados.\n"
        "* **Monocromático:** Modificación de la luminosidad (Lightness)."
    ),
    response_description="DTO estructurado con los colores resultantes y roles asignados."
)
async def generate_harmonic_palette(
    payload: CreatePaletteRequest = Body(
        ...,
        openapi_examples={
            "Prueba Complementaria": {
                "summary": "Rojo -> Complementario",
                "description": "Se envía un rojo puro (#FF0000) esperando un Cyan (#00FFFF) como complementario.",
                "value": {
                    "colors": ["#FF0000"],
                    "harmony": "complementario"
                }
            },
            "Prueba Triádica": {
                "summary": "Verde -> Triada",
                "description": "Se envía un verde puro (#00FF00) para obtener Azul y Rojo.",
                "value": {
                    "colors": ["#00FF00"],
                    "harmony": "triada"
                }
            }
        }
    )
):
    try:
        return se_generator.generate_palette(
            base_hex_colors=payload.colors,
            harmony=payload.harmony or HarmonyType.COMPLEMENTARY
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))