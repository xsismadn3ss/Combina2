"""Servicio de feedback: snapshot, persistencia y export CSV."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import TextIO

from sqlalchemy.orm import Session

from app.db.models import Feedback
from app.repositories import feedback as repo
from app.schemas.palette import FeedbackRequest

CSV_HEADERS = [
    "id",
    "created_at",
    "accepted",
    "score",
    "palette_type",
    "palette_json",
    "colors_hex",
    "image_sha256",
]

# Mitigación de CSV injection (Excel/Sheets ejecutan fórmulas con estos prefijos).
_INJECTION_PREFIXES = ("=", "+", "-", "@")


def record_feedback(session: Session, payload: FeedbackRequest) -> Feedback | None:
    """Persiste el feedback como snapshot JSONB de la paleta recibida."""
    snapshot = payload.palette.model_dump()
    return repo.insert_feedback(
        session,
        palette_json=snapshot,
        accepted=payload.accepted,
        score=payload.score,
        image_sha256=payload.imageSha256,
        source="api",
    )


def created_at_iso(value: datetime) -> str:
    """ISO-8601 UTC con sufijo ``Z`` (microsegundos)."""
    return repo.normalize_created_at(value).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def score_csv(value: float | None) -> str:
    return "" if value is None else f"{repo.normalize_score(value):.6f}"


def bool_csv(value: bool) -> str:
    return "true" if value else "false"


def palette_json_csv(value: dict | str) -> str:
    return repo.canonical_palette_json(value)


def colors_hex_csv(palette_json: dict | str | None) -> str:
    """HEX en orden separados por ``|``; vacío si el snapshot no los trae."""
    if not palette_json:
        return ""
    palette = json.loads(palette_json) if isinstance(palette_json, str) else palette_json
    colors = palette.get("colors") if isinstance(palette, dict) else None
    if not isinstance(colors, list):
        return ""
    values = [
        str(color.get("value", ""))
        for color in colors
        if isinstance(color, dict)
    ]
    return "|".join(value for value in values if value)


def csv_safe(value: str) -> str:
    """Prefija ``'`` a valores que un spreadsheet interpretaría como fórmula."""
    return f"'{value}" if value.startswith(_INJECTION_PREFIXES) else value


def feedback_csv_row(feedback: Feedback, *, include_sha256: bool = True) -> list[str]:
    values = [
        str(int(feedback.id or 0)),
        csv_safe(created_at_iso(feedback.created_at)),
        bool_csv(bool(feedback.accepted)),
        score_csv(feedback.score),
        csv_safe(feedback.palette_type or ""),
        csv_safe(palette_json_csv(feedback.palette_json)),
        csv_safe(colors_hex_csv(feedback.palette_json)),
        csv_safe(feedback.image_sha256 or "") if include_sha256 else "",
    ]
    return values


def _write_csv(writer: csv.writer, feedback: Feedback, include_sha256: bool) -> None:
    writer.writerow(feedback_csv_row(feedback, include_sha256=include_sha256))


def write_feedback_csv(
    session: Session,
    output: TextIO,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    accepted: bool | None = None,
    include_sha256: bool = True,
    batch_size: int = 500,
) -> int:
    """Escribe el CSV completo; devuelve el número de filas de datos."""
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(CSV_HEADERS)
    count = 0
    for feedback in repo.iter_feedback(
        session,
        start=start,
        end=end,
        accepted=accepted,
        batch_size=batch_size,
    ):
        _write_csv(writer, feedback, include_sha256)
        count += 1
    return count


def iter_feedback_csv_chunks(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    accepted: bool | None = None,
    include_sha256: bool = True,
    batch_size: int = 500,
) -> Iterator[bytes]:
    """CSV por lotes en UTF-8 (sin BOM), listo para ``StreamingResponse``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_HEADERS)
    yield _flush(buffer)

    for index, feedback in enumerate(
        repo.iter_feedback(
            session,
            start=start,
            end=end,
            accepted=accepted,
            batch_size=batch_size,
        ),
        start=1,
    ):
        _write_csv(writer, feedback, include_sha256)
        if index % batch_size == 0:
            yield _flush(buffer)
    yield _flush(buffer)


def _flush(buffer: io.StringIO) -> bytes:
    data = buffer.getvalue().encode("utf-8")
    buffer.seek(0)
    buffer.truncate(0)
    return data


def iter_csv_rows(rows: Iterable[dict]) -> Iterator[list[str]]:
    """Helper para el script de export (misma lógica que la API)."""
    yield list(CSV_HEADERS)
    for row in rows:
        yield [
            str(row["id"]),
            created_at_iso(row["created_at"]),
            bool_csv(bool(row["accepted"])),
            score_csv(row["score"]),
            csv_safe(row.get("palette_type") or ""),
            csv_safe(palette_json_csv(row["palette_json"])),
            csv_safe(colors_hex_csv(row["palette_json"])),
            csv_safe(row.get("image_sha256") or ""),
        ]
