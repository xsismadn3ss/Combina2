# Re-entrenamiento, promoción y rollback

Flujo completo para actualizar E/G/C con feedback real sin romper producción.
Los comandos asumen que estás en `backend/` con el venv de entrenamiento
(`.venv-tf`, Python 3.12 + TensorFlow). En Windows usa
`.venv-tf\Scripts\python.exe`; en Linux/macOS, `.venv-tf/bin/python`.

Regla de oro: **una versión nueva nunca sobrescribe a otra**. El registro
(`backend/models/manifest.json`, ver `docs/models.md`) conserva todas las
versiones y sus hashes; activar o hacer rollback solo mueve `active`/`previous`.

## 0. Requisitos

- Rama limpia y `git status` revisado. Los `.keras` **sí** se versionan.
- Postgres local arriba (`docker compose up -d postgres`) y bases migradas:
  `python scripts/db_bootstrap.py --target all --create --upgrade`.
- Gates de evaluación en verde antes de tocar producción.
- No borrar `classify-v1.keras`: es la versión de rollback de C.

## 1. Feedback → dataset

1. El frontend llama a `POST /api/v1/feedback` (`accepted`, `score`,
   `palette`, `imageSha256`). Cada fila queda en la tabla `feedback` del
   PostgreSQL de **runtime** (`DATABASE_URL`).
2. Exporta el feedback de producción y tráelo a la DB de entrenamiento local
   (`TRAINING_DATABASE_URL`):

   ```powershell
   # Con el backend público (o `railway connect Postgres` y el script local)
   curl.exe -fsS -H "Authorization: Bearer $env:FEEDBACK_EXPORT_TOKEN" `
     "https://<backend>/api/v1/feedback/export.csv" -o feedback.csv

   .venv-tf\Scripts\python.exe scripts\import_feedback_csv.py --input feedback.csv --dry-run
   .venv-tf\Scripts\python.exe scripts\import_feedback_csv.py --input feedback.csv --apply
   ```

   `created_at` se preserva, `source='import'` y el `fingerprint` evita
   duplicados si repites la importación. El CSV no contiene imágenes ni PII.

3. Refresca/mezcla fuentes si hace falta:

   ```powershell
   .venv-tf\Scripts\python.exe scripts\fetch_colourlovers.py
   .venv-tf\Scripts\python.exe scripts\fetch_colorpedia.py
   .venv-tf\Scripts\python.exe scripts\make_synthetic.py
   ```

4. Verifica el esquema (Alembic, no hay migraciones ad-hoc):

   ```powershell
   .venv-tf\Scripts\python.exe scripts\db_bootstrap.py --target training --upgrade
   .venv-tf\Scripts\python.exe -c "from sqlalchemy import inspect; from app.db.engine import training_engine; print([c['name'] for c in inspect(training_engine()).get_columns('training_runs')])"
   ```

## 2. Re-entrenar una versión nueva

Cada script elige por defecto la siguiente `vN` libre:

```powershell
# E / G / C
.venv-tf\Scripts\python.exe scripts\train_extract.py
.venv-tf\Scripts\python.exe scripts\train_generate.py
.venv-tf\Scripts\python.exe scripts\train_classify.py
```

- `--version 2` o `--version v2` fija `extract-v2.keras`, `generate-v2.keras`,
  etc. Si el destino ya existe el script **falla antes de entrenar**.
- Sin `--version` se auto-incrementa a la siguiente versión libre.
- `--out <ruta>` prevalece sobre `--version` por compatibilidad; se registra el
  nombre real del archivo. Evítalo salvo casos puntuales.
- `--force` sobrescribe la versión destino y también permite registrar aunque
  no se alcance el umbral de entrenamiento. **No lo uses para pisar una versión
  buena**; úsalo solo en entornos desechables.
- Al terminar, el script llama a `model_registry.register_version`, que:
  calcula sha256, guarda `createdAt`, `gitSha`, `trainingRunId`, versiones de
  TF/Keras, métricas (y `params`), y deja la nueva versión como `active`
  moviendo la anterior a `previous`.
- **Splits disjuntos (DB + sintético):** `classify_split_palettes` carga
  las paletas de la DB de entrenamiento una sola vez, baraja con semilla fija
  y reparte train/val/test en buckets disjuntos (`seed % 3`). El relleno
  sintético usa una semilla distinta por split. Ninguna paleta de DB cae en dos
  splits; si la DB está vacía el set sintético es idéntico al histórico.

La activación del script es **provisional**: producción aún no está validada
hasta pasar los gates del paso 4. Si fallan, haz rollback inmediatamente
(paso 5).

## 3. Calibrar (solo si corresponde)

```powershell
# C: temperatura en validación
.venv-tf\Scripts\python.exe scripts\calibrate.py --slot classify --source reconstruction
# G: Platt solo si AUC >= 0.55
.venv-tf\Scripts\python.exe scripts\calibrate.py --slot generate --source auto
# ambos
.venv-tf\Scripts\python.exe scripts\calibrate.py --slot all
```

- Nunca se calibra con el split de test.
- G **no se calibra** si su AUC de score es < 0.55
  (`eval/thresholds.json → calibration.require_auc_min`). Se deja
  `calibrated:false` con el motivo.
- C v1 (40 dims legacy) se conserva sin calibrar (`reason:"legacy_40d"`); el
  rollback mantiene su semántica cruda.

## 4. Evaluar y exigir gates

```powershell
.venv-tf\Scripts\python.exe -m eval.run_eval --strict --check-manifest
```

- `--strict` sale con código 1 si falla cualquier gate de
  `backend/eval/thresholds.json`, hay drift de métricas contra el manifiesto o
  falta TensorFlow.
- `--check-manifest` compara las métricas medidas con las registradas
  (tolerancia 10%).
- Estado esperado hoy: **12/12 gates PASS** y **26 métricas** verificadas.

Opcional para dejar evidencia:

```powershell
.venv-tf\Scripts\python.exe -m eval.run_eval --strict --check-manifest --json eval-report.json --md eval-report.md
```

## 5. Promover o hacer rollback

Con los gates en verde, la versión nueva queda promovida (ya es `active`).
Puedes verla y verificarla:

```powershell
.venv-tf\Scripts\python.exe scripts\manage_models.py list
.venv-tf\Scripts\python.exe scripts\manage_models.py verify
```

Si un gate falla o el comportamiento empeora, vuelve a la anterior:

```powershell
# 1) Ver qué hará, sin escribir
.venv-tf\Scripts\python.exe scripts\manage_models.py rollback --slot classify --dry-run

# 2) Ejecutar (intercambia active <-> previous; es reversible)
.venv-tf\Scripts\python.exe scripts\manage_models.py rollback --slot classify

# 3) Re-evaluar
.venv-tf\Scripts\python.exe -m eval.run_eval --strict --check-manifest
```

`rollback` exige que el slot tenga `previous` y que el archivo exista. Nunca
borra la versión rechazada: puede volver a promoverse con `activate`.

Activación manual de una versión ya presente en `backend/models/`:

```powershell
.venv-tf\Scripts\python.exe scripts\manage_models.py activate --slot classify --file classify-v1.keras --note "rollback manual"
```

### Caché de modelos en caliente

La inferencia indexa la caché Keras por `slot:archivo`, así que un rollback no
devuelve el modelo viejo. Aun así, en producción se recomienda **reiniciar el
proceso** (o llamar a `app.services.inference.clear_model_cache()`) tras
promover o hacer rollback para liberar el modelo anterior de memoria.

### Caché de features de evaluación

`.eval-cache/` (gitignored) guarda las features E→C por
`(split, n, seed, versión/hash de E)`. La clave incluye el nombre del `.keras`
resuelto y los 8 primeros hex de su sha256 (`extract_cache_tag`), así que
reentrenar o activar otra E **invalida automáticamente** la caché; no hay que
borrarla a mano ni se reutilizan features de la E anterior.

## 6. Qué NO hacer

- **No sobrescribir** un `.keras` existente (ni con `--force` en producción).
- **No borrar** versiones ni entradas de `versions`/`files`: rompe rollback y
  `manage_models.py verify`.
- **No calibrar G** mientras su AUC < 0.55: se empeora el ranking sin base.
- **No calibrar C con test**: la temperatura se ajusta solo en validación.
- **No promover** una versión con gates en rojo.
- **No editar `manifest.json` a mano**; usa `scripts/train_*.py` o
  `scripts/manage_models.py`.
- **No borrar `classify-v1.keras`**: es la garantía de rollback de C.
