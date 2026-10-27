"""Tablas de entrenamiento (datos locales, nunca en producción)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CHAR,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.db.models.runtime import Feedback as Feedback  # re-export: misma tabla


class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, Identity(), primary_key=True),
    )
    sha256: str = Field(sa_column=Column(CHAR(64), nullable=False, unique=True))
    width: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    height: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    source: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class ExtractedColor(SQLModel, table=True):
    __tablename__ = "extracted_colors"

    image_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("images.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    rank: int = Field(sa_column=Column(Integer, primary_key=True))
    hex: str = Field(sa_column=Column(Text, nullable=False))
    l: float = Field(sa_column=Column(Float, nullable=False))
    a: float = Field(sa_column=Column(Float, nullable=False))
    b: float = Field(sa_column=Column(Float, nullable=False))
    weight: float = Field(sa_column=Column(Float, nullable=False))


class Palette(SQLModel, table=True):
    __tablename__ = "palettes"

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, Identity(), primary_key=True),
    )
    key: str = Field(sa_column=Column(CHAR(64), nullable=False, unique=True))
    source: str = Field(sa_column=Column(Text, nullable=False))
    rating: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    harmony_label: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    __table_args__ = (Index("ix_palettes_source", "source"),)


class PaletteColor(SQLModel, table=True):
    __tablename__ = "palette_colors"

    palette_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("palettes.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    position: int = Field(sa_column=Column(Integer, primary_key=True))
    hex: str = Field(sa_column=Column(Text, nullable=False))
    l: float = Field(sa_column=Column(Float, nullable=False))
    a: float = Field(sa_column=Column(Float, nullable=False))
    b: float = Field(sa_column=Column(Float, nullable=False))


class TrainingRun(SQLModel, table=True):
    __tablename__ = "training_runs"

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, Identity(), primary_key=True),
    )
    model: str = Field(sa_column=Column(Text, nullable=False))
    git_sha: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    params_json: dict | None = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    metrics_json: dict | None = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    keras_path: str = Field(sa_column=Column(Text, nullable=False))
    tf_version: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    keras_version: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    calibration_json: dict | None = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    __table_args__ = (Index("ix_training_runs_model", "model"),)


__all__ = [
    "ExtractedColor",
    "Feedback",
    "Image",
    "Palette",
    "PaletteColor",
    "TrainingRun",
]
