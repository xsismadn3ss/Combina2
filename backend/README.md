# Combina2 backend (Fase 6 — issue #22)

FastAPI + TensorFlow. Estado: redes **E/G/C activas** en `models/manifest.json`
(`extract-v1`, `generate-v1`, `classify-v2`), con calibración de C y registro de
versiones/rollback en `/api/v1/models`. Si TF o un `.keras` no están disponibles,
la API responde por el **fallback clásico** (`fallback: true`) sin caerse.

## Correr

```bash
uv venv .venv && uv pip install -e .          # runtime (sin TF): API + fallback
uvicorn app.main:app --reload                # desde backend/
pytest                                        # tests de contrato
```

`TRAINING_DB` (opcional): ruta del sqlite. Defecto: `backend/training.db`
(versionado como evidencia de runs/feedback; excepción de `.gitignore`).

## Venvs: runtime vs entrenamiento

Solo hay Python 3.14 en el sistema y TensorFlow publica `2.22.0rc0` para 3.14
(pre-release); para entrenar con TF estable se usa un venv separado con 3.12:

| venv | Python | Paquetes | Uso |
|---|---|---|---|
| `backend/.venv` | 3.14 | runtime **sin TF** (`-e .`) | API, `pytest` (el path con red E activa nunca importa sklearn ni exige TF: sin TF hay fallback) |
| `backend/.venv-tf` | 3.12 | `tensorflow-cpu` + `numpy` + `pillow` + `scikit-learn` (+ deps de API para tests) | **entrenamiento y eval**: `scripts/train_*.py` y `python -m eval.run_eval` |

```bash
uv venv backend/.venv-tf --python 3.12
uv pip install --python backend/.venv-tf/bin/python -r backend/requirements-ml.txt
backend/.venv-tf/bin/python scripts/train_extract.py --epochs 50  # red E
```

`requirements-ml.txt` fija TF 2.21.0/Keras 3.15.1 y usa **Python 3.12**: TF
estable no publica wheel para 3.14 (por eso el runtime 3.14 corre sin TF y los
tests de modelo hacen skip).

(`pyproject.toml` exige `>=3.14`, así que en `.venv-tf` se instalan los
paquetes sueltos sin `-e .`; los tests corren igual por `pythonpath = ["."]`.)
Ambos venvs están gitignored; los `.keras` de `backend/models/` **sí** se
versionan.

## Endpoints (`/api/v1`)

| Método | Ruta | Motor hoy |
|---|---|---|
| GET | `/health` | readiness |
| GET | `/harmony`, `/colors/roles` | enums |
| GET | `/models` | registro read-only desde `models/manifest.json` (sin TF) |
| POST | `/palette/extract` | red E; fallback KMeans clásico si E no está disponible |
| POST | `/palette/generate` | red G + condicionamiento armónico con `topK`; fallback HSL |
| POST | `/palette/identify` | pipeline E→C con calibración; fallback `predominant` |
| POST | `/feedback` | persiste en sqlite |

Contratos: [`../docs/api.md`](../docs/api.md). Modelos: [`models/README.md`](models/README.md).

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
  de 80 paletas; incluye calibración aplicada (ECE/NLL antes/después y gates
  `calibration/ece` y `calibration/nll` solo si `calibrated:true`).
- `--check-manifest`: compara lo medido con `models/manifest.json` (tolerancia
  10%; bajo `--strict` un drift falla).
- Sin TF: slots `skipped`, exit 0 salvo `--strict`.

## Calibración (B2)

```bash
# desde backend/ con .venv-tf (TF 2.21 / Keras 3.15)
.venv-tf/Scripts/python.exe scripts/calibrate.py --slot all
.venv-tf/Scripts/python.exe scripts/calibrate.py --slot classify --dry-run
.venv-tf/Scripts/python.exe scripts/calibrate.py --slot generate --source auto
```

- `classify`: ajusta la temperatura **solo con el split de validación**
  (120 paletas, seed 10042), imprime ECE/NLL antes/después (val y test como
  referencia) y escribe `calibration` en el manifiesto. Con `--dry-run` no
  escribe nada.
- `generate`: usa feedback real si hay ≥ 50 filas; si no, el proxy de
  reconstrucción (40 paletas seed 99, buena = CIE76 ≤ 14). Rechaza Platt si
  AUC < `calibration.require_auc_min` (0.55) y deja `calibrated:false` con
  motivo; no invierte el score.
- `/identify` expone `confidenceCalibrated`, `needsRetry` y `suggestion`.
  `IDENTIFY_MIN_CONFIDENCE` (default **0.70**) controla cuándo se pide otra
  foto; justificación risk-coverage en `../docs/api.md`.

CI (`.github/workflows/ci.yml`): `backend-runtime` (3.14, sin TF),
`backend-ml` (3.12, tests completos + eval estricta + artefactos) y `web`
(Node 22: lint/typecheck/build).
