"""Repositorio de ``training_runs`` (DB de entrenamiento)."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import TrainingRun


def create_training_run(
    session: Session,
    *,
    model: str,
    keras_path: str,
    git_sha: str | None = None,
    params_json: dict | None = None,
    metrics_json: dict | None = None,
    tf_version: str | None = None,
    keras_version: str | None = None,
    calibration_json: dict | None = None,
    run_id: int | None = None,
) -> TrainingRun:
    """Registra un run. ``run_id`` explícito solo se usa al migrar SQLite."""
    run = TrainingRun(
        model=model,
        keras_path=keras_path,
        git_sha=git_sha,
        params_json=params_json,
        metrics_json=metrics_json,
        tf_version=tf_version,
        keras_version=keras_version,
        calibration_json=calibration_json,
        created_at=utcnow(),
    )
    if run_id is not None:
        run.id = int(run_id)
    session.add(run)
    session.flush()
    return run


def delete_training_run(session: Session, run_id: int) -> int:
    result = session.execute(delete(TrainingRun).where(TrainingRun.id == int(run_id)))
    return int(result.rowcount or 0)


def get_training_run(session: Session, run_id: int) -> TrainingRun | None:
    return session.get(TrainingRun, int(run_id))


def list_training_runs(session: Session, model: str | None = None) -> list[TrainingRun]:
    stmt = select(TrainingRun).order_by(TrainingRun.id)
    if model is not None:
        stmt = stmt.where(TrainingRun.model == model)
    return list(session.execute(stmt).scalars())
