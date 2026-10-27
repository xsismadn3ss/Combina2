# Combina2

Combina 2 o más colores para generar una paleta. Identifica paletas de colores tomando una foto.

## Documentación

- [Backend](./backend/README.md)
- [Datos y PostgreSQL](./docs/data.md)
- [Re-entrenamiento, promoción y rollback](./docs/retraining.md)
- [Despliegue en Railway](./docs/deployment.md)
- [API](./docs/api.md)

## Estado (Fase 7)

- Redes **E/G/C** activas desde `backend/models/manifest.json` con registro
  versionado (`active`/`previous`/`versions`/`files`, sha256 por archivo).
- C usa `classify-v2.keras` (54 dims, calibrada); `classify-v1.keras` se
  conserva para rollback. G sigue sin calibrar (AUC < 0.55). E/G/C tienen
  fallback clásico si falta el modelo.
- **Datos en PostgreSQL**: `feedback` de runtime en `DATABASE_URL`; datos de
  entrenamiento en `TRAINING_DATABASE_URL` (solo local). Esquema único con
  Alembic; `backend/training.db` queda como evidencia histórica deprecada.
- **Producción (Railway)**: backend Docker con Python 3.12 + TensorFlow,
  Postgres gestionado solo para feedback, frontend Next same-origin y
  healthcheck `GET /ready`. Ver [`docs/deployment.md`](./docs/deployment.md).

## Quickstart local

```bash
cp .env.example .env              # credenciales de desarrollo
docker compose up -d postgres     # Postgres 17 + volumen persistente

cd backend
# runtime (sin TF)
uv venv .venv && uv pip install -e .
.venv/bin/python scripts/db_bootstrap.py --target all --create --upgrade
.venv/bin/python -m pytest tests -q
```

En Windows: `backend\.venv\Scripts\python.exe`.
`pytest` necesita Postgres arriba (no hay SQLite): las fixtures crean y migran
`combina2_test`/`combina2_training_test` con Alembic.

## CI / evaluación local

Rutas en minúscula (el repo usa `backend/`; imprescindible en Linux/macOS,
donde el filesystem distingue mayúsculas):

```bash
# tests (ambos venvs) — requieren Postgres local
backend/.venv/bin/python -m pytest backend/tests -q
backend/.venv-tf/bin/python -m pytest backend/tests -q

cd backend
../backend/.venv-tf/bin/python -m eval.run_eval --strict --check-manifest
../backend/.venv-tf/bin/python scripts/manage_models.py verify
```

En Windows/PowerShell los venvs están en `.venv\Scripts\python.exe` y las rutas
admiten mayúsculas, pero se mantiene `backend\` en minúscula por consistencia:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q
backend\.venv-tf\Scripts\python.exe -m pytest backend\tests -q
cd backend
..\backend\.venv-tf\Scripts\python.exe -m eval.run_eval --strict --check-manifest
..\backend\.venv-tf\Scripts\python.exe scripts\manage_models.py verify
```

`--strict --check-manifest` exige los gates de `backend/eval/thresholds.json`
y que las métricas medidas coincidan con el manifiesto (tolerancia 10%).

## Ejecutar con Docker

Desde la raíz: `docker compose up -d --build` y abre <http://localhost:3000>.
El servicio `postgres` guarda los datos en el volumen `combina2-pgdata`.
Antes del primer arranque del backend en Docker, migra la DB de runtime:

```bash
docker compose run --rm backend alembic upgrade head
```

Consulta registros con `docker compose logs -f`; detén el stack con
`docker compose down`. El backend publica <http://localhost:8000> y su
healthcheck espera a `GET /ready` (los 3 modelos cargados). La imagen **no
incluye `backend/training.db`**: en runtime solo usa `models/*.keras` +
`manifest.json` (+ Postgres para feedback).
