"""Feedback de runtime + export CSV protegido (issue #28)."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import APICOnfig, feedback_export_token
from app.db.session import get_session, runtime_session
from app.repositories.feedback import count_feedback
from app.schemas.palette import FeedbackRequest, FeedbackResponse
from app.services.feedback import iter_feedback_csv_chunks, record_feedback

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{APICOnfig.prefix}", tags=["feedback"])

RETRY_AFTER_SECONDS = 30
CSV_FILENAME = "feedback.csv"


def _require_export_token(authorization: str | None) -> None:
    """Bearer token obligatorio.

    Sin ``FEEDBACK_EXPORT_TOKEN`` el endpoint se comporta como inexistente
    (404) para no filtrar la existencia de datos; token incorrecto → 401.
    """
    expected = feedback_export_token()
    if expected is None:
        raise HTTPException(status_code=404, detail="Not Found")
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Token de export inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/feedback",
    description=(
        "Registra si el usuario aceptó una paleta. Alimenta el re-entrenamiento "
        "de G/C (tabla `feedback` en PostgreSQL)."
    ),
    response_description="Confirmación del registro.",
)
def feedback(
    payload: FeedbackRequest,
    session: Session = Depends(get_session),
) -> FeedbackResponse:
    try:
        record_feedback(session, payload)
        # Commit explícito: un fallo al confirmar (DB caída/constraint) también
        # debe responder 503, no un 500 desde el teardown de la dependencia.
        session.commit()
    except Exception as exc:  # noqa: BLE001 - DB caída => 503, sin filtrar DSN
        logger.error("feedback insert failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Feedback no disponible temporalmente",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        ) from None
    return FeedbackResponse(ok=True)


@router.get(
    "/feedback/export.csv",
    description=(
        "Exporta el feedback como CSV (UTF-8). Requiere "
        "`Authorization: Bearer $FEEDBACK_EXPORT_TOKEN`."
    ),
    response_class=StreamingResponse,
)
def export_feedback_csv(
    authorization: str | None = Header(default=None),
    start: datetime | None = Query(default=None, alias="from"),
    end: datetime | None = Query(default=None, alias="to"),
    accepted: bool | None = Query(default=None),
    include_sha256: bool = Query(default=True),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    _require_export_token(authorization)
    # Comprobación temprana: si la DB no responde, el 503 sale antes de
    # empezar el stream (si fallara a mitad, ya no se puede cambiar el status).
    try:
        count_feedback(session, start=start, end=end, accepted=accepted)
    except Exception as exc:  # noqa: BLE001
        logger.error("feedback export failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Export no disponible temporalmente",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        ) from None

    def generate():
        with runtime_session() as stream_session:
            yield from iter_feedback_csv_chunks(
                stream_session,
                start=start,
                end=end,
                accepted=accepted,
                include_sha256=include_sha256,
            )

    return StreamingResponse(
        generate(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{CSV_FILENAME}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
