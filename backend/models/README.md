# Modelos entrenados (persistencia en el repositorio)

Esta carpeta **sí se versiona en git**: es el entregable de producción. Cada red ocupa
un slot (`extract` | `generate` | `classify`) y el backend sirve el `.keras` marcado
como `active` en [`manifest.json`](manifest.json). Los 3 slots `extract`/`generate`/`classify` ya tienen `active` (Fase 4 — `extract-v1.keras`, `generate-v1.keras`, `classify-v1.keras`); si un `.keras` faltara, la API responde con el motor clásico y `fallback: true`.

## Política

- Formato: Keras v3 (`*.keras`), uno por versión: `<slot>-v< N >.keras` (ej. `generate-v1.keras`).
- Registrar = copiar el `.keras` aquí + añadirlo a `versions` + mover `active` + insertar
  fila en `training_runs` (ver `docs/data.md`). Rollback = mover `active` atrás, sin borrar.
- Nuestras redes son MLPs pequeñas (< 5 MB): caben en git normal. Solo si un futuro
  backbone convolucional supera ~50 MB se evaluará Git LFS; no es el caso hoy.
- Lo que **no** va al repo: `training.db` (datos de entrenamiento, gitignored) ni los
  checkpoints intermedios de `scripts/` (van a `training_runs.keras_path` temporal).
