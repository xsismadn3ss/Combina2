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
│   │   ├── ml/
│   │   │   ├── color_spaces.py     # rgb↔hsl↔lab + normalización a [0,1]/[-1,1]
│   │   │   ├── datasets.py         # loaders HF/API/sintético → sqlite
│   │   │   └── losses.py           # MSE-Lab + aproximación CIEDE2000
│   │   └── db/sqlite.py            # acceso a training.db
│   ├── scripts/
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

Ignorados en git: `backend/training.db*`, `backend/*.db`, `__pycache__/`, `web/node_modules/`,
`web/.next/`. Los `.keras` **sí** se versionan (son el entregable de producción).

## 3. Flujos

### Historia 1 — Generar paleta
`web /generar` → `POST /extract` (E: foto → top-10 + pesos) → usuario selecciona 1..n →
`POST /generate {colors, topK}` (G: candidatas rankeadas con score) → feedback 👍/👎 →
`POST /feedback` (cierra el loop a sqlite).

### Historia 2 — Identificar paleta
`web /identificar` → `POST /identify` (foto → E extrae 10 colores → C clasifica armonía →
`{type, confidence, colors}`).

### Entrenamiento (offline, nunca en request)
fetch/cura → sqlite → `scripts/train_*.py` → `backend/models/<nombre>-vX.keras` +
entrada en `manifest.json` → el backend carga la versión `active`.

## 4. Decisiones (ADR resumido)

| # | Decisión | Motivo |
|---|----------|--------|
| 1 | 100% neuronal en inferencia (sin sklearn en request) | Requisito del issue; E es soft-KMeans/entrenable en TF destilado de KMeans |
| 2 | Entrenar en **CIELAB**, no RGB | Espacio perceptualmente uniforme; loss ΔE/CIEDE2000 correlaciona con juicio humano |
| 3 | Clasificador **MLP sobre paleta extraída** (no CNN sobre imagen) | Decisión del dueño; liviano, explicable, suficiente para 4 clases de armonía |
| 4 | Dataset público + sintético bootstrap | COLOURLovers/Color-Pedia dan escala; el motor HSL actual da etiquetas día-0 |
| 5 | Next.js reemplaza Blazor (no convivencia) | Decisión del dueño; prototipado rápido con shadcn |
| 6 | Fallback a reglas si falta el `.keras` | La API nunca se cae por un modelo ausente; responde con motor clásico + `fallback: true` |
