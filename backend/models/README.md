# Modelos entrenados (persistencia en el repositorio)

Esta carpeta **sí se versiona en git**: es el entregable de producción. Cada red ocupa
un slot (`extract` | `generate` | `classify`) y el backend sirve el `.keras` resuelto
desde [`manifest.json`](manifest.json). Hoy: `extract-v1.keras`, `generate-v1.keras` y
`classify-v2.keras` activos; `classify-v1.keras` se conserva como `previous` para
rollback. Si un `.keras` faltara o su hash no coincidiera, la inferencia intenta
`previous` y, si tampoco, la API responde con el motor clásico y `fallback: true`.

## Política de versiones

- Formato: Keras v3 (`*.keras`), uno por versión: `<slot>-v<N>.keras`
  (ej. `classify-v3.keras`).
- **Nunca se sobrescribe ni se borra una versión.** Un re-entreno crea siempre un
  archivo nuevo (`--version N` o la siguiente `vN` libre). Los scripts fallan si el
  destino existe salvo `--force`, reservado para entornos desechables.
- Registrar una versión (lo hacen `scripts/train_*.py` vía
  `app.services.model_registry.register_version`) implica:
  1. copiar el `.keras` a esta carpeta,
  2. calcular `sha256` y metadatos (`createdAt`, `gitSha`, `trainingRunId`,
     `tfVersion`, `kerasVersion`, `metrics`, `calibration`),
  3. añadir a `versions`, mover el `active` anterior a `previous` y activar la nueva.
- Rollback = intercambiar `active` y `previous`; no se borra nada.
- Nuestras redes son MLPs pequeñas (< 5 MB): caben en git normal. Solo si un futuro
  backbone convolucional supera ~50 MB se evaluará Git LFS; no es el caso hoy.
- `backend/training.db` **sí se versiona como evidencia** de runs y feedback
  (excepción `!backend/training.db` en `.gitignore`); no guarda pesos. Lo que
  **no** va al repo son los checkpoints intermedios de `scripts/` (se guardan
  como `.tmp-*.keras` y se registran/copian al final) ni las cachés
  (`.eval-cache/`).

## Herramientas

Desde `backend/` (no requiere TensorFlow):

```powershell
python scripts\manage_models.py list
python scripts\manage_models.py verify [--slot classify] [--json]
python scripts\manage_models.py activate --slot classify --file classify-v1.keras --note "..."
python scripts\manage_models.py rollback --slot classify [--dry-run]
```

`verify` devuelve exit 1 si falta un archivo o su sha256 no coincide.
El flujo completo de re-entrenamiento/calibración/promoción está en
[`docs/retraining.md`](../../docs/retraining.md).

## Caché

La inferencia indexa los modelos Keras por `slot:archivo`, así que un rollback no
sirve el modelo viejo. En producción, reinicia el proceso o llama a
`app.services.inference.clear_model_cache()` tras activar/rollback.
