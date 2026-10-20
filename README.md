# Combina2

Combina 2 o más colores para generar una paleta. Identífica paletas de colores tomando una foto.

## Documentación

- [Backend](./backend/README.md)
- [Re-entrenamiento, promoción y rollback](./docs/retraining.md)

## Estado (Fase 6)

- Redes **E/G/C** activas desde `backend/models/manifest.json` con registro
  versionado (`active`/`previous`/`versions`/`files`, sha256 por archivo).
- C usa `classify-v2.keras` (54 dims, calibrada); `classify-v1.keras` se
  conserva para rollback. G sigue sin calibrar (AUC < 0.55). E/G/C tienen
  fallback clásico si falta el modelo.
- Registro y rollback se gestionan con `backend/scripts/manage_models.py`; el
  flujo completo está en [`docs/retraining.md`](./docs/retraining.md).

## CI / evaluación local

Rutas en minúscula (el repo usa `backend/`; imprescindible en Linux/macOS,
donde el filesystem distingue mayúsculas):

```bash
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
Consulta registros con `docker compose logs -f`; detén el stack con
`docker compose down`. El backend publica <http://localhost:8000>. La base
existente `backend/training.db` se monta directamente y conserva los datos;
el archivo debe existir antes de arrancar.
