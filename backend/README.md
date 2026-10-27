# Combina2 backend (Fase 7 — issues #22/#28)

FastAPI + TensorFlow. Estado: redes **E/G/C activas** en `models/manifest.json`
(`extract-v1`, `generate-v1`, `classify-v2`), con calibración de C y registro de
versiones/rollback en `/api/v1/models`. El feedback y los datos de
entrenamiento viven en **PostgreSQL**. Si TF o un `.keras` no están disponibles,
la API responde por el **fallback clásico** (`fallback: true`) sin caerse.

## Quickstart (Postgres local + venv)

```bash
# 1) Desde la RAÍZ del repo: Postgres en Docker (.env vive en la raíz)
cp .env.example .env          # solo la primera vez
docker compose up -d postgres

# 2) Runtime sin TF (desde backend/)
cd backend
uv venv .venv && uv pip install -e .
.venv/bin/python scripts/db_bootstrap.py --target all --create --upgrade
.venv/bin/python -m pytest tests -q          # requiere Postgres real
.venv/bin/uvicorn app.main:app --reload
```

En Windows los venvs están en `backend\.venv\Scripts\python.exe` y
`backend\.venv-tf\Scripts\python.exe`.

Variables (ver `.env.example`):

| Variable | Uso |
|---|---|
| `DATABASE_URL` | DB de runtime (feedback). Default local `combina2`. |
| `TRAINING_DATABASE_URL` | DB de entrenamiento (paletas, runs, feedback importado). **No se define en producción.** |
| `FEEDBACK_EXPORT_TOKEN` | Bearer de `GET /api/v1/feedback/export.csv` (vacío ⇒ 404). |
| `DB_CONNECT_TIMEOUT` | Timeout de conexión en segundos (default 5). |
| `IDENTIFY_MIN_CONFIDENCE` | Umbral de `/identify` (default 0.70). |

`backend/training.db` (SQLite) se conserva como evidencia histórica y está
**deprecado**: ningún código de `app/` lo usa. Migración one-shot:
`python scripts/migrate_sqlite_to_postgres.py --sqlite training.db --apply`.

## Venvs: runtime vs entrenamiento

El runtime se puede ejecutar en Python 3.12+ (`pyproject.toml`:
`requires-python = ">=3.12"`). En Docker/producción se usa **Python 3.12 con
TensorFlow** para cargar las redes E/G/C; localmente hay dos venvs:

| venv | Python | Paquetes | Uso |
|---|---|---|---|
| `backend/.venv` | 3.14 | runtime **sin TF** (`-e .`) | API, `pytest` (sin TF hay fallback) |
| `backend/.venv-tf` | 3.12 | `tensorflow-cpu` + deps de API y DB | **entrenamiento, eval e inferencia real**: `scripts/train_*.py`, `python -m eval.run_eval` |

```bash
uv venv backend/.venv-tf --python 3.12
uv pip install --python backend/.venv-tf/bin/python -r backend/requirements-ml.txt
backend/.venv-tf/bin/python scripts/train_extract.py --epochs 50  # red E
```

`requirements-ml.txt` fija TF 2.21.0/Keras 3.15.1 y usa **Python 3.12**: TF
estable no publica wheel para 3.14. El `Dockerfile` de producción usa también
Python 3.12 e instala el extra `.[ml]` (`tensorflow-cpu==2.21.0`,
`keras==3.15.1`), por lo que en Railway las redes cargan de verdad; si TF o un
`.keras` faltan, la API sigue respondiendo por el fallback clásico.

`psycopg[binary]` 3.3.x publica wheels para 3.12/3.13/3.14, así que la capa de
datos es idéntica en los dos venvs y en CI. Ambos venvs están gitignored; los
`.keras` de `backend/models/` **sí** se versionan.

## Endpoints (`/api/v1`)

| Método | Ruta | Motor hoy |
|---|---|---|
| GET | `/health` | liveness (200, sin cargar modelos ni DB) |
| GET | `/ready` | readiness real de E/G/C (200/503; `storage: models_only` + `database` informativo) |
| GET | `/harmony`, `/colors/roles` | enums |
| GET | `/models` | registro read-only desde `models/manifest.json` (sin TF) |
| POST | `/palette/extract` | red E; fallback KMeans clásico si E no está disponible |
| POST | `/palette/generate` | red G + condicionamiento armónico con `topK`; fallback HSL |
| POST | `/palette/identify` | pipeline E→C con calibración; fallback `predominant` |
| POST | `/feedback` | persiste snapshot en PostgreSQL (503 si la DB cae) |
| GET | `/feedback/export.csv` | CSV protegido con Bearer (`FEEDBACK_EXPORT_TOKEN`) |

Contratos: [`../docs/api.md`](../docs/api.md). Datos y migraciones:
[`../docs/data.md`](../docs/data.md). Modelos: [`models/README.md`](models/README.md).
Despliegue en Railway: [`../docs/deployment.md`](../docs/deployment.md).

## Base de datos

- SQLModel + SQLAlchemy 2 sync + psycopg 3; engines con `pool_pre_ping`.
- Un único conjunto de migraciones Alembic crea el esquema completo en las dos
  bases (`alembic upgrade head`, advisory lock, idempotente).
- `python scripts/db_bootstrap.py --target all --create --upgrade` crea y migra
  local/CI. En Railway el pre-deploy es `alembic upgrade head` con
  `DATABASE_URL`.
- Export/import/purga:

  ```bash
  python scripts/export_feedback_csv.py --output feedback.csv
  python scripts/import_feedback_csv.py --input feedback.csv --dry-run
  python scripts/import_feedback_csv.py --input feedback.csv --apply
  python scripts/purge_feedback.py --before 2025-01-01 --apply
  ```

## Evaluación (E/G/C) y gates

Fuente única de umbrales: `eval/thresholds.json`. Línea base reproducible:
`eval/baseline.json`.

```bash
# desde backend/ (con .venv-tf: TF 2.21 / Keras 3.15)
python -m eval.run_eval                         # reporte a stdout
python -m eval.run_eval --slot classify         # un solo slot
python -m eval.run_eval --strict --check-manifest \
  --json eval-report.json --md eval-report.md   # CI: exit 1 si falla un gate
```

- `extract`: camino real `inference.extract_image_colors` vs KMeans clásico
  (10 mosaicos, seeds 5000..5009).
- `generate`: predicción cruda de G (40 paletas seed 99) con CIEDE2000 real.
- `classify`: pipeline E→C con `inference.identify_image` sobre el split test
  de 80 paletas; incluye calibración aplicada y gates `calibration/ece`/`nll`.
- `--check-manifest`: compara lo medido con `models/manifest.json` (tolerancia
  10%; bajo `--strict` un drift falla).
- Sin TF: slots `skipped`, exit 0 salvo `--strict`.
- Con la DB de entrenamiento vacía, los sets usan el relleno sintético
  determinista (la línea base histórica).

## Calibración (B2)

```bash
.venv-tf/Scripts/python.exe scripts/calibrate.py --slot all
.venv-tf/Scripts/python.exe scripts/calibrate.py --slot classify --dry-run
.venv-tf/Scripts/python.exe scripts/calibrate.py --slot generate --source auto
```

- `classify`: ajusta la temperatura **solo con el split de validación**
  (120 paletas, seed 10042) y escribe `calibration` en el manifiesto.
- `generate`: usa feedback real de la DB de entrenamiento si hay ≥ 50 filas; si
  no, el proxy de reconstrucción (40 paletas seed 99). Rechaza Platt si
  AUC < 0.55 y no invierte el score.
- `/identify` expone `confidenceCalibrated`, `needsRetry` y `suggestion`.

CI (`.github/workflows/ci.yml`): `backend-runtime` (3.14, sin TF),
`backend-ml` (3.12, tests completos + eval estricta + artefactos),
`persistence` (Postgres de compose sobrevive a un restart) y `web`
(Node 22: lint/typecheck/build). Ambos jobs de backend usan un service
container `postgres:17` real y `scripts/db_bootstrap.py`.
