from fastapi import FastAPI

from src.routes.pallete import router as pallete_router
from src.routes.colors import router as colors_router
from src.routes.harmoy import router as harmony_router

app = FastAPI(
    title="Combina2 Sistemas Expertos API",
    version="1.0.0",
    description="Backend FastAPI para la generación e identificación armónica de colores."
)

app.include_router(pallete_router)
app.include_router(colors_router)
app.include_router(harmony_router)
