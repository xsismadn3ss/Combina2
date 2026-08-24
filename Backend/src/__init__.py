from fastapi import FastAPI
from src.api.palettes import router as palettes_router

app = FastAPI(
    title="Combina2 Sistemas Expertos API",
    version="1.0.0",
    description="Backend FastAPI para la generación e identificación armónica de colores."
)

# ¡Esta es la línea clave que le falta a tu app para mostrar el endpoint!
app.include_router(palettes_router)