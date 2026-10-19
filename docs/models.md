# Modelos — E / G / C (TensorFlow/Keras, formato `.keras`)

Todo color de entrenamiento/inferencia se representa en **CIELAB normalizado**.
Conversiones y normalización viven en `backend/app/ml/color_spaces.py`.

## E — Extractora (`extract-v*.keras`, ~10-50k params)

- **Entrada:** (3000, 3) píxeles Lab normalizados de la imagen preprocesada.
- **Arquitectura:** MLP por píxel 3→64→64 (tanh) + capa *soft-assignment* a K=10
  (KMeans diferenciable: distancias → softmax con temperatura) + rama de saliencia
  (atención ligera por píxel que pondera el conteo).
- **Salida:** (10, 3) centroides Lab + (10,) pesos, ordenados por peso.
- **Loss:** reconstrucción MSE-Lab píxel→centroide asignado + λ·entropía (bins nítidos).
- **Entrenamiento:** destilación del KMeans actual (`MAX_CLUSTERS=5`→10, `MAX_SIZE=300`,
  `RANDOM_STATE=42`) sobre imágenes cura + Color-Pedia.
- **Eval de paridad:** ΔE medio vs KMeans original sobre set fijo; objetivo ΔE < 5.

## G — Generadora (`generate-v*.keras`, ~50-200k params)

- **Entrada:** (10, 3) paleta Lab + máscara multi-hot (10,) de seleccionados +
  embedding de armonía (4→8, opcional).
- **Arquitectura:** aplanado 10·3+10+8=48 → MLP 48→128→128 (ReLU, dropout 0.1) →
  cabeza paleta (15) + cabeza compatibilidad (1, sigmoide estilo O'Donovan).
- **Salida:** N paletas de 5 colores Lab + score (beam/muestreo con temperaturas
  distintas para diversidad); el servicio convierte a HEX y rankea.
- **Loss:** MSE-Lab + μ·CIEDE2000 aproximado diferenciable + BCE del head de rating.
- **Datos:** COLOURLovers + Color-Pedia + sintético HSL. **Eval:** CIEDE2000 vs
  ground truth (referencia paper WikiArt+CGAN: ~23.5 en 5 colores).

## C — Clasificadora (`classify-v*.keras`, ~10k params)

- **Entrada:** (10, 3) Lab + (10,) pesos = 40 dims (lo que devuelve E).
- **Arquitectura:** MLP 40→64→32 (ReLU) → softmax 4
  (`complementario|triada|analogo|monocromatica`).
- **Loss:** cross-entropy + label smoothing 0.05. **Eval:** accuracy top-1 +
  matriz de confusión; objetivo ≥ 0.85 en sintético, ≥ 0.70 en curado.

## Registro (`backend/models/manifest.json`)

```json
{
  "extract":   { "active": "extract-v1.keras",   "versions": ["extract-v1.keras"] },
  "generate":  { "active": "generate-v1.keras",  "versions": ["generate-v1.keras"] },
  "classify":  { "active": "classify-v1.keras",  "versions": ["classify-v1.keras"] }
}
```

`inference.py` carga perezosamente el `active`; si el archivo falta, responde con el
motor clásico y `"fallback": true` en el DTO (la API nunca falla por un modelo ausente).
Cada `train_*.py` inserta una fila en `training_runs` con `git_sha`, hiperparámetros y
métricas.
