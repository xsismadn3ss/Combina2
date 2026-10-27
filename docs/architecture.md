# Arquitectura de software

## 1. Visión

Foto → top-10 colores → (a) el usuario elige 1+ y recibe paletas candidatas rankeadas,
o (b) el sistema clasifica qué armonía describe mejor la foto. Toda la inferencia de
color vive en grafos TensorFlow; la extracción clásica (KMeans/sklearn) solo existe
como **maestra de destilación** en entrenamiento, nunca en inferencia.

## 2. Layout del monorepo (objetivo)

```
/
├── backend/
│   ├── app/
│   │   ├── main.py                 # factory FastAPI, /health
│   │   ├── api/routes/
│   │   │   ├── extract.py          # POST /api/v1/palette/extract  (red E)
│   │   │   ├── generate.py         # POST /api/v1/palette/generate (red G)
│   │   │   └── identify.py         # POST /api/v1/palette/identify (E→C)
│   │   ├── schemas/palette.py      # DTOs (port de main + weight/score/confidence)
│   │   ├── services/inference.py   # carga perezosa de .keras, pre/post-proceso
│   │   ├── repositories/           # feedback + training_runs (SQLAlchemy)
│   │   ├── ml/
│   │   │   ├── color_spaces.py     # rgb↔hsl↔lab + normalización a [0,1]/[-1,1]
│   │   │   ├── datasets.py         # loaders HF/API/sintético → PostgreSQL
│   │   │   └── losses.py           # MSE-Lab + aproximación CIEDE2000
│   │   ├── db/
│   │   │   ├── engine.py           # engines psycopg (sync, pool_pre_ping)
│   │   │   ├── session.py          # runtime_session/training_session/get_session
│   │   │   └── models/             # SQLModel: feedback + tablas de entrenamiento
│   │   └── services/feedback.py    # snapshot + CSV export
│   ├── alembic.ini
│   ├── migrations/                 # Alembic (un esquema para las dos DBs)
│   ├── scripts/
│   │   ├── db_bootstrap.py         # crear/migrar bases (local y CI)
│   │   ├── migrate_sqlite_to_postgres.py
│   │   ├── export_feedback_csv.py
│   │   ├── import_feedback_csv.py
│   │   ├── purge_feedback.py
│   │   ├── fetch_colourlovers.py
│   │   ├── fetch_colorpedia.py
│   │   ├── make_synthetic.py
│   │   ├── train_extract.py        # red E
│   │   ├── train_generate.py       # red G
│   │   └── train_classify.py       # red C
│   ├── models/                     # *.keras + manifest.json (SÍ en repo)
│   └── tests/                      # contrato API + paridad ΔE + accuracy
├── web/                            # Next.js App Router + shadcn (reemplaza Web/ Blazor)
└── docs/                           # esta carpeta
```

Ignorados en git: `.env`, `backend/*.db` (salvo `backend/training.db`, conservado
como evidencia histórica y **deprecado**), `__pycache__/`, `web/node_modules/`,
`web/.next/`. Los `.keras` **sí** se versionan (son el entregable de producción).

## 3. Flujos

### Historia 1 — Generar paleta
`web /generar` → `POST /extract` (E: foto → top-10 + pesos) → usuario selecciona 1..n →
`POST /generate {colors, topK}` (G: candidatas rankeadas con score) → feedback 👍/👎 →
`POST /feedback` (snapshot a PostgreSQL, `DATABASE_URL`).

### Historia 2 — Identificar paleta
`web /identificar` → `POST /identify` (foto → E extrae 10 colores → C clasifica armonía →
`{type, confidence, colors}`).

### Entrenamiento (offline, nunca en request)
fetch/cura → PostgreSQL de entrenamiento (`TRAINING_DATABASE_URL`) →
`scripts/train_*.py` → `backend/models/<nombre>-vX.keras` + entrada en
`manifest.json` → el backend carga la versión `active`.

### Feedback de producción → entrenamiento
El runtime solo escribe `feedback` en `DATABASE_URL`. Para reentrenar se exporta
(`GET /feedback/export.csv` con Bearer) y se importa a la DB local de
entrenamiento con `scripts/import_feedback_csv.py` (fingerprint idempotente).
La DB de entrenamiento **no** existe en producción.

## 4. Decisiones (ADR resumido)

| # | Decisión | Motivo |
|---|----------|--------|
| 1 | 100% neuronal en inferencia (sin sklearn en request) | Requisito del issue; E es soft-KMeans/entrenable en TF destilado de KMeans |
| 2 | Entrenar en **CIELAB**, no RGB | Espacio perceptualmente uniforme; loss ΔE/CIEDE2000 correlaciona con juicio humano |
| 3 | Clasificador **MLP sobre paleta extraída** (no CNN sobre imagen) | Decisión del dueño; liviano, explicable, suficiente para 4 clases de armonía |
| 4 | Dataset público + sintético bootstrap | COLOURLovers/Color-Pedia dan escala; el motor HSL actual da etiquetas día-0 |
| 5 | Next.js reemplaza Blazor (no convivencia) | Decisión del dueño; prototipado rápido con shadcn |
| 6 | Fallback a reglas si falta el `.keras` | La API nunca se cae por un modelo ausente; responde con motor clásico + `fallback: true` |
| 7 | PostgreSQL + SQLModel/SQLAlchemy 2 sync + psycopg 3 | Las dos bases (runtime y entrenamiento) comparten esquema/migraciones Alembic; sync encaja con FastAPI en threadpool y psycopg 3 tiene wheels en 3.12–3.14 |
| 8 | La inferencia no depende de la DB | `/health`, `/ready` y `/palette/*` siguen funcionando si Postgres cae; solo el feedback/export devuelven error |
| 9 | Feedback como snapshot JSONB sin FK a `palettes` | El feedback de producción puede referirse a paletas que no existen en la DB local; permite exportar/importar entre bases sin dependencias |
