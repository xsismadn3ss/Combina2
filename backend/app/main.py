from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.extract import router as extract_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.generate import router as generate_router
from app.api.routes.identify import router as identify_router
from app.api.routes.meta import colors_router, harmony_router


def create_app() -> FastAPI:
    application = FastAPI(
        title="Combina2 API",
        version="0.1.0",
        description=(
            "Paletas de color con redes neuronales TensorFlow "
            "(fallback clásico mientras no haya modelos activos)."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(extract_router)
    application.include_router(generate_router)
    application.include_router(identify_router)
    application.include_router(harmony_router)
    application.include_router(colors_router)
    application.include_router(feedback_router)

    @application.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
