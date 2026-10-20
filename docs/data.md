# Datos — sqlite + datasets

`training.db` vive en `backend/` y **sí se versiona como evidencia** de los runs
de entrenamiento y del feedback recogido (excepción explícita en `.gitignore`:
`*.db` ignorados salvo `!backend/training.db`). Nunca guarda pesos: esos van en
`backend/models/`. Es evidencia de trazabilidad (filas de `training_runs`,
versiones TF/Keras, calibración) y, además, los scripts de entrenamiento leen
de él las paletas etiquetadas de armonía.

## 1. DDL

```sql
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY,
  sha256 TEXT UNIQUE NOT NULL,
  width INTEGER, height INTEGER,
  source TEXT NOT NULL,              -- colourlovers|colorpedia|wikiart|synthetic|user
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extracted_colors(
  image_id INTEGER NOT NULL REFERENCES images(id),
  rank INTEGER NOT NULL,             -- 0..9 por ponderancia
  hex TEXT NOT NULL,
  l REAL NOT NULL, a REAL NOT NULL, b REAL NOT NULL,
  weight REAL NOT NULL,              -- fracción de píxeles / atención
  PRIMARY KEY(image_id, rank)
);

CREATE TABLE IF NOT EXISTS palettes(
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  rating REAL,                       -- likes/5, hearts normalizados o NULL
  harmony_label TEXT,                -- complementario|triada|analogo|monocromatica|NULL
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS palette_colors(
  palette_id INTEGER NOT NULL REFERENCES palettes(id),
  position INTEGER NOT NULL,
  hex TEXT NOT NULL,
  l REAL NOT NULL, a REAL NOT NULL, b REAL NOT NULL,
  PRIMARY KEY(palette_id, position)
);

CREATE TABLE IF NOT EXISTS training_runs(
  id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,               -- extract|generate|classify
  git_sha TEXT, params_json TEXT,
  metrics_json TEXT,                 -- ej. {"val_ciede2000": 13.1}
  keras_path TEXT NOT NULL,          -- backend/models/<slot>-vN.keras
  tf_version TEXT,                   -- B3: versión de TensorFlow del run
  keras_version TEXT,                -- B3: versión de Keras del run
  calibration_json TEXT,             -- B3: calibración aplicada o NULL
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY,
  image_id INTEGER REFERENCES images(id),
  palette_id INTEGER REFERENCES palettes(id),
  accepted INTEGER,                  -- 1/0
  score REAL,
  palette_json TEXT,                 -- paleta enviada por /feedback (JSON)
  image_sha256 TEXT,                 -- sha256 de la foto, si el cliente lo envía
  created_at TEXT DEFAULT (datetime('now'))
);
```

## 1.1 Migración B3 (`training_runs`)

`app/db/sqlite.py` ejecuta una migración **idempotente** al abrir la conexión:
si faltan, añade `tf_version`, `keras_version` y `calibration_json` con
`ALTER TABLE ... ADD COLUMN`. La `training.db` existente se conserva sin
pérdida de filas; `calibration_json` es nullable (los runs nuevos de
entrenamiento no calibran; la calibración se registra aparte en el manifiesto
y puede copiarse aquí cuando aplique).

## 2. Fuentes (orden de prioridad)

1. **`boltuix/color-pedia` (HF, ~50k, Parquet, MIT)** — arranque inmediato, sin scraping.
   Campos: RGB/HEX + nombre + emoción. Uso: pre-entrenar G y head de compatibilidad.
2. **COLOURLovers API `GET /api/palettes/top`** — top-1000 × 8 `hueOption`
   (red, orange, yellow, green, aqua, blue, violet, fuchsia) ≈ 7k paletas de 5 colores
   con ranking (técnica del paper n-Grams 2023). Uso: G + C.
3. **Sintético del motor HSL actual** (`main:Backend/src/services/palette_generator.py`) —
   generar (base, armonía) → paleta + ruido Lab; etiqueta perfecta día-0 para C y
   bootstrap de G. Si (1)+(2) no rinden, este es el plan B completo.
4. **WikiArt subset** — **descartado en Fase 6 (ADR).** E se entrena por
   destilación de KMeans sobre mosaicos sintéticos + paletas de `training.db`;
   el resultado híbrido `HardKMeans` ya cumple ΔE76 < 5 y no necesita imágenes
   de arte. Criterio para revisarlo: solo si una futura E amortizada (sin
   KMeans en inferencia) no alcanza los gates de paridad con mosaicos
   sintéticos, o si se aborda un dominio de fotos reales que los mosaicos no
   representen. En ese caso se documentará la licencia del subset y el
   preproceso/dedup por `sha256` antes de incorporarlo.

## 3. Preproceso común (`datasets.py`)

Resize lado mayor 300px · RGBA→RGB sobre blanco · submuestreo 3.000 px ·
RGB→CIELAB (vía `color_spaces.py`) · normalización L∈[0,1], a/b∈[-1,1] aprox.
Dedup por `sha256`. Split 80/10/10 estratificado por `harmony_label`/`source`.
