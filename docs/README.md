# Combina2 — Documentación de arquitectura (refactor TensorFlow)

> Rama de trabajo: `16-combina2` · Epic: issue #16.
> Stack objetivo: **Python** (FastAPI + TensorFlow + PostgreSQL) · **TypeScript** (Next.js + shadcn).
> Estado anterior (rama `main`): SE1 reglas HSL (`palette_generator.py`) + SE2 KMeans
> (`palette_identifier.py`). Este refactor los reemplaza por 3 redes neuronales.

## Índice

- [`architecture.md`](./architecture.md) — visión, monorepo, flujos historia 1 y 2, decisiones.
- [`data.md`](./data.md) — PostgreSQL (runtime + entrenamiento), Alembic, datasets público + sintético, retención.
- [`models.md`](./models.md) — redes E / G / C: entradas, salidas, loss, entrenamiento, registro/versiones/rollback.
- [`retraining.md`](./retraining.md) — flujo feedback → dataset → re-entrenar → calibrar → eval → promover/rollback.
- [`api.md`](./api.md) — contratos REST estables para el frontend.
- [`frontend.md`](./frontend.md) — app Next.js que reemplaza `Web/` (Blazor).
- [`deployment.md`](./deployment.md) — despliegue en Railway (web público + backend privado + Postgres para feedback, same-origin, Alembic, backups, export, `/ready`, rollback y smoke E2E).
- [`research.md`](./research.md) — bibliografía que sustenta enseñarle color a la red.

## Supuestos abiertos (por confirmar)

1. Extracción fija en **K=10** colores, paletas generadas de **5 colores** (estándar COLOURLovers).
2. Clasificador con **4 clases**: `complementario | triada | analogo | monocromatica`.
3. Artefactos de producción en `backend/models/*.keras` + `manifest.json` (versionados en repo).
4. `POST /generate` devuelve **ranking multi-armonía con score**, `harmony` pasa a ser opcional/filtro.
