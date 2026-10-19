# Datos — sqlite + datasets

`training.db` vive en `backend/` y **no se versiona** (ver `.gitignore`). Guarda datos de
entrenamiento y feedback, nunca pesos (esos van en `backend/models/`).

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
  metrics_json TEXT,                 -- ej. {"val_caddy2000": 23.5}
  keras_path TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY,
  image_id INTEGER REFERENCES images(id),
  palette_id INTEGER REFERENCES palettes(id),
  accepted INTEGER,                  -- 1/0
  score REAL,
  created_at TEXT DEFAULT (datetime('now'))
);
```

## 2. Fuentes (orden de prioridad)

1. **`boltuix/color-pedia` (HF, ~50k, Parquet, MIT)** — arranque inmediato, sin scraping.
   Campos: RGB/HEX + nombre + emoción. Uso: pre-entrenar G y head de compatibilidad.
2. **COLOURLovers API `GET /api/palettes/top`** — top-1000 × 8 `hueOption`
   (red, orange, yellow, green, aqua, blue, violet, fuchsia) ≈ 7k paletas de 5 colores
   con ranking (técnica del paper n-Grams 2023). Uso: G + C.
3. **Sintético del motor HSL actual** (`main:Backend/src/services/palette_generator.py`) —
   generar (base, armonía) → paleta + ruido Lab; etiqueta perfecta día-0 para C y
   bootstrap de G. Si (1)+(2) no rinden, este es el plan B completo.
4. **WikiArt subset** (opcional, fase 6) — solo si E necesita más variedad de imágenes.

## 3. Preproceso común (`datasets.py`)

Resize lado mayor 300px · RGBA→RGB sobre blanco · submuestreo 3.000 px ·
RGB→CIELAB (vía `color_spaces.py`) · normalización L∈[0,1], a/b∈[-1,1] aprox.
Dedup por `sha256`. Split 80/10/10 estratificado por `harmony_label`/`source`.
