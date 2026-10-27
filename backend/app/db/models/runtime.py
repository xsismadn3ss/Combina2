"""Tablas del runtime (feedback de producción)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Identity,
    Index,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class Feedback(SQLModel, table=True):
    """Feedback de usuario (snapshot autosuficiente de una paleta).

    ``fingerprint`` es el SHA-256 estable del contenido de la fila y permite
    importaciones idempotentes entre bases (``ON CONFLICT DO NOTHING``).
    """

    __tablename__ = "feedback"

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, Identity(), primary_key=True),
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    accepted: bool = Field(sa_column=Column(Boolean, nullable=False))
    score: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    palette_type: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    palette_json: dict = Field(sa_column=Column(JSONB, nullable=False))
    image_sha256: str | None = Field(
        default=None, sa_column=Column(CHAR(64), nullable=True)
    )
    fingerprint: str = Field(
        sa_column=Column(CHAR(64), nullable=False, unique=True)
    )
    source: str = Field(
        default="api",
        sa_column=Column(Text, nullable=False, server_default="api"),
    )

    __table_args__ = (
        CheckConstraint(
            "score IS NULL OR (score >= 0 AND score <= 1)",
            name="score_range",
        ),
        CheckConstraint(
            "image_sha256 IS NULL OR image_sha256 ~ '^[0-9a-f]{64}$'",
            name="image_sha256_hex",
        ),
        Index("ix_feedback_created_at", "created_at"),
        Index("ix_feedback_accepted", "accepted"),
        Index("ix_feedback_accepted_created_at", "accepted", "created_at"),
    )
