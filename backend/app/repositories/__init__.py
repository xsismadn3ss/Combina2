"""Repositorios: acceso a datos con ``Session`` (sin SQL crudo salvo upserts)."""

from app.repositories import feedback, training

__all__ = ["feedback", "training"]
