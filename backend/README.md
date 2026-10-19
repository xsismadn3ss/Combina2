# Combina2 backend (Fase 1 — issue #17)

FastAPI + (pronto) TensorFlow. Estado: esqueletos con **fallback clásico** porteado de
`main`; ninguna red entrenada todavía (`models/manifest.json` tiene `active: null`).

## Correr

```bash
uv venv .venv && uv pip install -e ".[ml]"   # sin [ml] también funciona (fallback)
uvicorn app.main:app --reload                # desde backend/
pytest                                        # tests de contrato
```

`TRAINING_DB` (opcional): ruta del sqlite. Defecto: `backend/training.db` (gitignored).

## Endpoints (`/api/v1`)

| Método | Ruta | Motor hoy |
|---|---|---|
| GET | `/health` | — |
| GET | `/harmony`, `/colors/roles` | enums |
| POST | `/palette/extract` | clásico KMeans, `fallback: true` |
| POST | `/palette/generate` | clásico HSL, `fallback: true` (1 paleta; ranking en Fase 4) |
| POST | `/palette/identify` | clásico + tipo `predominant`, `fallback: true` |
| POST | `/feedback` | persiste en sqlite |

Contratos: [`../docs/api.md`](../docs/api.md). Modelos: [`models/README.md`](models/README.md).
