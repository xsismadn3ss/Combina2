"""Esquema inicial PostgreSQL (runtime + entrenamiento).

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_images"),
        sa.UniqueConstraint("sha256", name="uq_images_sha256"),
    )

    op.create_table(
        "palettes",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("key", sa.CHAR(length=64), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("harmony_label", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_palettes"),
        sa.UniqueConstraint("key", name="uq_palettes_key"),
    )
    op.create_index("ix_palettes_source", "palettes", ["source"])

    op.create_table(
        "extracted_colors",
        sa.Column("image_id", sa.BigInteger(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("hex", sa.Text(), nullable=False),
        sa.Column("l", sa.Float(), nullable=False),
        sa.Column("a", sa.Float(), nullable=False),
        sa.Column("b", sa.Float(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["images.id"],
            name="fk_extracted_colors_image_id_images",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("image_id", "rank", name="pk_extracted_colors"),
    )

    op.create_table(
        "palette_colors",
        sa.Column("palette_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("hex", sa.Text(), nullable=False),
        sa.Column("l", sa.Float(), nullable=False),
        sa.Column("a", sa.Float(), nullable=False),
        sa.Column("b", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["palette_id"],
            ["palettes.id"],
            name="fk_palette_colors_palette_id_palettes",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("palette_id", "position", name="pk_palette_colors"),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("git_sha", sa.Text(), nullable=True),
        sa.Column("params_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("keras_path", sa.Text(), nullable=False),
        sa.Column("tf_version", sa.Text(), nullable=True),
        sa.Column("keras_version", sa.Text(), nullable=True),
        sa.Column(
            "calibration_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_training_runs"),
    )
    op.create_index("ix_training_runs_model", "training_runs", ["model"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("palette_type", sa.Text(), nullable=True),
        sa.Column(
            "palette_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("image_sha256", sa.CHAR(length=64), nullable=True),
        sa.Column("fingerprint", sa.CHAR(length=64), nullable=False),
        sa.Column("source", sa.Text(), server_default="api", nullable=False),
        sa.CheckConstraint(
            "score IS NULL OR (score >= 0 AND score <= 1)",
            name="score_range",
        ),
        sa.CheckConstraint(
            "image_sha256 IS NULL OR image_sha256 ~ '^[0-9a-f]{64}$'",
            name="image_sha256_hex",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feedback"),
        sa.UniqueConstraint("fingerprint", name="uq_feedback_fingerprint"),
    )
    op.create_index("ix_feedback_created_at", "feedback", ["created_at"])
    op.create_index("ix_feedback_accepted", "feedback", ["accepted"])
    op.create_index(
        "ix_feedback_accepted_created_at", "feedback", ["accepted", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_accepted_created_at", table_name="feedback")
    op.drop_index("ix_feedback_accepted", table_name="feedback")
    op.drop_index("ix_feedback_created_at", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_training_runs_model", table_name="training_runs")
    op.drop_table("training_runs")
    op.drop_table("palette_colors")
    op.drop_table("extracted_colors")
    op.drop_index("ix_palettes_source", table_name="palettes")
    op.drop_table("palettes")
    op.drop_table("images")
