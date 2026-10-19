from fastapi import APIRouter, HTTPException

from app.config import APICOnfig
from app.db.sqlite import insert_feedback
from app.schemas.palette import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix=f"{APICOnfig.prefix}", tags=["feedback"])


@router.post(
    "/feedback",
    description=(
        "Registra si el usuario aceptó una paleta. Alimenta el re-entrenamiento "
        "de G/C (tabla `feedback` en training.db)."
    ),
    response_description="Confirmación del registro.",
)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        insert_feedback(
            palette_json=payload.palette.model_dump_json(),
            accepted=payload.accepted,
            score=payload.score,
            image_sha256=payload.imageSha256,
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return FeedbackResponse(ok=True)
