"""Repositorio de ``feedback`` (compartido runtime/entrenamiento)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import Feedback

SCORE_DECIMALS = 6


def canonical_palette_json(palette_json: dict | str) -> str:
    """JSON compacto, estable (claves ordenadas) y UTF-8 sin escapar."""
    if isinstance(palette_json, str):
        palette_json = json.loads(palette_json)
    return json.dumps(
        palette_json, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def normalize_score(score: float | None) -> float | None:
    """Redondea a 6 decimales (misma precisión que el CSV exportado).

    Así el ``fingerprint`` de una fila exportada e importada coincide aunque el
    float original tuviera más precisión.
    """
    if score is None:
        return None
    return round(float(score), SCORE_DECIMALS)


def normalize_created_at(value: datetime | None = None) -> datetime:
    """UTC *aware* con precisión de microsegundos (cancelable entre bases)."""
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def feedback_fingerprint(
    *,
    created_at: datetime,
    accepted: bool,
    score: float | None,
    palette_json: dict | str,
    image_sha256: str | None,
) -> str:
    """SHA-256 del contenido de la fila (idempotencia al importar)."""
    created = normalize_created_at(created_at).isoformat(timespec="microseconds")
    score_text = "" if score is None else f"{normalize_score(score):.6f}"
    payload = "|".join(
        [
            created,
            "1" if accepted else "0",
            score_text,
            canonical_palette_json(palette_json),
            (image_sha256 or "").lower(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def insert_feedback(
    session: Session,
    *,
    palette_json: dict | str,
    accepted: bool,
    score: float | None,
    image_sha256: str | None,
    source: str = "api",
    created_at: datetime | None = None,
) -> Feedback | None:
    """Inserta un feedback; ``None`` si ya existía (mismo fingerprint).

    ``palette_json`` es el snapshot completo ``payload.palette.model_dump()``.
    ``palette_type`` se denormaliza para filtros baratos. No hay FK a
    ``palettes``: el snapshot puede no existir en la DB de entrenamiento.
    """
    snapshot = (
        json.loads(palette_json) if isinstance(palette_json, str) else palette_json
    )
    created = normalize_created_at(created_at)
    normalized_score = normalize_score(score)
    fingerprint = feedback_fingerprint(
        created_at=created,
        accepted=accepted,
        score=normalized_score,
        palette_json=snapshot,
        image_sha256=image_sha256,
    )
    palette_type = snapshot.get("type") if isinstance(snapshot, dict) else None
    stmt = (
        pg_insert(Feedback)
        .values(
            created_at=created,
            accepted=bool(accepted),
            score=normalized_score,
            palette_type=palette_type,
            palette_json=snapshot,
            image_sha256=image_sha256.lower() if image_sha256 else None,
            fingerprint=fingerprint,
            source=source,
        )
        .on_conflict_do_nothing(index_elements=["fingerprint"])
        .returning(Feedback.id)
    )
    row_id = session.execute(stmt).scalar_one_or_none()
    if row_id is None:
        return None
    session.flush()
    return session.get(Feedback, int(row_id))


def get_feedback(session: Session, feedback_id: int) -> Feedback | None:
    return session.get(Feedback, feedback_id)


def iter_feedback(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    accepted: bool | None = None,
    batch_size: int = 500,
) -> Iterator[Feedback]:
    """Itera feedback ordenado por id con cursor por lotes (streaming)."""
    stmt = select(Feedback).order_by(Feedback.id)
    if start is not None:
        stmt = stmt.where(Feedback.created_at >= normalize_created_at(start))
    if end is not None:
        stmt = stmt.where(Feedback.created_at <= normalize_created_at(end))
    if accepted is not None:
        stmt = stmt.where(Feedback.accepted.is_(bool(accepted)))
    stmt = stmt.execution_options(yield_per=batch_size)
    yield from session.execute(stmt).scalars()


def count_feedback(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    accepted: bool | None = None,
) -> int:
    stmt = select(func.count()).select_from(Feedback)
    if start is not None:
        stmt = stmt.where(Feedback.created_at >= normalize_created_at(start))
    if end is not None:
        stmt = stmt.where(Feedback.created_at <= normalize_created_at(end))
    if accepted is not None:
        stmt = stmt.where(Feedback.accepted.is_(bool(accepted)))
    return int(session.execute(stmt).scalar_one())


def delete_before(session: Session, cutoff: datetime) -> int:
    """Borra feedback anterior a ``cutoff``; devuelve filas borradas."""
    from sqlalchemy import delete

    result = session.execute(
        delete(Feedback).where(Feedback.created_at < normalize_created_at(cutoff))
    )
    return int(result.rowcount or 0)


def count_before(session: Session, cutoff: datetime) -> int:
    """Filas de feedback anteriores a ``cutoff`` (para dry-run de purga)."""
    stmt = (
        select(func.count())
        .select_from(Feedback)
        .where(Feedback.created_at < normalize_created_at(cutoff))
    )
    return int(session.execute(stmt).scalar_one())


def row_payload(feedback: Feedback, *, include_sha256: bool = True) -> dict[str, Any]:
    """Valores de una fila listos para CSV (sin imágenes ni rutas)."""
    return {
        "id": int(feedback.id or 0),
        "created_at": feedback.created_at,
        "accepted": bool(feedback.accepted),
        "score": feedback.score,
        "palette_type": feedback.palette_type,
        "palette_json": feedback.palette_json,
        "image_sha256": feedback.image_sha256 if include_sha256 else None,
    }
