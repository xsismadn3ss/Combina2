# Combina2 backend (Fase 1 — issue #17)

FastAPI + (pronto) TensorFlow. Estado: esqueletos con **fallback clásico** porteado de
`main`; ninguna red entrenada todavía (`models/manifest.json` tiene `active: null`).

## Correr

```bash
uv venv .venv && uv pip install -e .          # runtime (sin TF): API + fallback
uvicorn app.main:app --reload                # desde backend/
pytest                                        # tests de contrato
```

`TRAINING_DB` (opcional): ruta del sqlite. Defecto: `backend/training.db` (gitignored).

## Venvs: runtime vs entrenamiento (Fase 3)

Solo hay Python 3.14 en el sistema y TensorFlow publica `2.22.0rc0` para 3.14
(pre-release); para entrenar con TF estable se usa un venv separado con 3.12:

| venv | Python | Paquetes | Uso |
|---|---|---|---|
| `backend/.venv` | 3.14 | runtime **sin TF** (`-e .`) | API, `pytest` (el path con red E activa nunca importa sklearn ni exige TF: sin TF hay fallback) |
| `backend/.venv-tf` | 3.12 | `tensorflow` + `numpy` + `pillow` + `scikit-learn` (+ deps de API para tests) | **solo entrenamiento**: `scripts/train_extract.py`; también sirve para correr la paridad ΔE con el `.keras` real |

```bash
uv venv backend/.venv-tf --python 3.12
uv pip install --python backend/.venv-tf/bin/python \
  "tensorflow>=2.19" "numpy>=1.26" pillow scikit-learn \
  fastapi httpx pydantic pytest python-multipart uvicorn
backend/.venv-tf/bin/python scripts/train_extract.py --epochs 50  # red E
```

(`pyproject.toml` exige `>=3.14`, así que en `.venv-tf` se instalan los
paquetes sueltos sin `-e .`; los tests corren igual por `pythonpath = ["."]`.)
Ambos venvs están gitignored; los `.keras` de `backend/models/` **sí** se
versionan.

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
