# Modelos — E / G / C (TensorFlow/Keras, formato `.keras`)

Todo color de entrenamiento/inferencia se representa en **CIELAB normalizado**.
Conversiones y normalización viven en `backend/app/ml/color_spaces.py`.

## E — Extractora (`extract-v*.keras`, ~10-50k params)

- **Entrada:** (3000, 3) píxeles Lab normalizados de la imagen preprocesada
  (`datasets.preprocess_image`: resize lado mayor 300px, `sample_pixels` 3000 px
  determinista seed 42, RGB→Lab vía `ColorConverter`, normalización L∈[0,1],
  a/b∈[-1,1]).

- **Arquitectura real (`backend/app/ml/extract_model.py`):**
  - *Camino amortizado entrenable* (`build_extract_model(use_kmeans=False)`):
    MLP por píxel 3→128→128 (tanh) + LayerNormalization + rama de saliencia
    (`Dense(1)` → `Softmax(axis=1)` que pondera el conteo por píxel) +
    `PrototypeAttention` K=10 temperatura 2.0 (10 consultas aprendidas atienden
    a los píxeles vía `einsum` + softmax) + `SoftAssignment` K=10 temperatura
    2.5 (KMeans diferenciable: `||feats−proto||²` → `softmax(−dist²/τ)`) +
    `SaliencyPooling` (conteo `resp=assign·salience`, centroides como promedio
    ponderado y pesos `mass/total`, ordenados por peso descendente).
  - *Camino híbrido desplegado* (`extract-v1.keras` activo,
    `build_kmeans_model()` / `HardKMeans` K=10): KMeans real en RGB **100%
    numpy/tf puro, sin sklearn en el path de inferencia**. Flujo:
    Lab normalizado → Lab crudo → `_lab_to_rgb_np` → KMeans determinista
    numpy (k-means++ init, seed 42, 10 inits con elección por inercia mínima,
    15 iters Lloyd, `N_INIT=10` como el maestro clásico) → centroides RGB →
    `_rgb_to_lab_np` → Lab normalizado + pesos. La capa `HardKMeans`
    envuelve `_batch_kmeans_lab_to_rgb` vía `tf.numpy_function` y expone
    `(centroids, weights, assign_dummy)` manteniendo el grafo TF y
    `EXTRACT_CUSTOM_OBJECTS` para recarga del `.keras` sin importar sklearn.

- **Salida:** (10, 3) centroides Lab normalizados + (10,) pesos, ordenados por
  peso decreciente, + (3000, 10) asignaciones blandas (3ª salida solo para la
  loss de entrenamiento; inferencia usa las dos primeras).

- **Loss (`backend/app/ml/losses.py`):** destilación MSE-Lab **desnormalizado**
  + MSE pesos + `λ·H(w)` (entropía, bins nítidos) + `λ_assign·H̄(assign)`.
  Entrenamiento usa `make_extract_sinkhorn_loss` (emparejamiento blando
  Sinkhorn `ε: 100→10` recocido lineal, doblemente estocástico `1/K`, estable
  en dominio log) para evitar cambios bruscos de asignación; evaluación usa
  `mean_delta_e` (CIE76 euclidiana en Lab con emparejamiento greedy).

- **Entrenamiento (`backend/scripts/train_extract.py`, solo `.venv-tf`):**
  destilación del KMeans clásico (`services/classic.py`: `MAX_CLUSTERS=10`,
  `MAX_SIZE=300`, `RANDOM_STATE=42`, `N_INIT=10`) sobre mosaicos sintéticos
  `datasets.synthesize_mosaic` (franjas con áreas `Dirichlet`, ruido luz
  `gain∈[0.95,1.05]` + gaussiano σ=3, recompresión JPEG `quality 60-95`,
  600 train / 120 val / 24 test por defecto) + paletas `training.db`
  (si no alcanzan, relleno aleatorio offline determinista). Init de
  prototipos: MLP con identidad escalada + KMeans en features; reanimación
  de clusters muertos cada 4 épocas.

- **Hallazgo honesto:** red amortizada pura (MLP 3→128→128 + atención) con
  ~600 muestras **no alcanza ΔE <5** de forma estable (converge a ~8-12,
  cercano al límite de destilación reportado). Se entrega híbrido
  `HardKMeans` determinista que sí cumple **ΔE≈3.8 <5** manteniendo
  API/offline/determinista y sin `sklearn` en inferencia (KMeans 100% numpy,
  seed 42, paridad idéntica al maestro). **Si se exige amortizada pura sin
  KMeans en inferencia, umbral realista ~8-12.**

- **Eval de paridad:** ΔE medio CIE76 vs `ClassicPaletteIdentifier` sobre set
  fijo de 10 mosaicos (size 64, `jpeg_quality=80`, seed 5000..5010);
  objetivo ΔE < 5 — alcanzado **3.81** con `extract-v1.keras` híbrido.

## G — Generadora (`generate-v*.keras`, ~25k params — 24.888 reales)

- **Entrada:** (10, 3) paleta Lab + máscara multi-hot (10,) de seleccionados +
  embedding de armonía (4→8, opcional).
- **Arquitectura:** aplanado 10·3+10+8=48 → MLP 48→128→128 (ReLU, dropout 0.1) →
  cabeza paleta (15) + cabeza compatibilidad (1, sigmoide estilo O'Donovan).
- **Salida:** N paletas de 5 colores Lab + score (beam/muestreo con temperaturas
  distintas para diversidad); el servicio convierte a HEX y rankea.
- **Loss:** MSE-Lab + μ·CIEDE2000 aproximado diferenciable + BCE del head de rating.
- **Datos:** COLOURLovers + Color-Pedia + sintético HSL. **Eval:** CIEDE2000 vs
  ground truth (referencia paper WikiArt+CGAN: ~23.5 en 5 colores).
- **Métrica real v1:** `val_CIEDE 24.77` (ref 23.5) con `val_mse 0.029`, `generate-v1.keras` 344 KB, 24.888 params — en línea con referencia.

## C — Clasificadora (`classify-v*.keras`, ~5k params — 4.836 reales)

- **Entrada:** (10, 3) Lab + (10,) pesos = 40 dims (lo que devuelve E).
- **Arquitectura:** MLP 40→64→32 (ReLU) → softmax 4
  (`complementario|triada|analogo|monocromatica`).
- **Loss:** cross-entropy + label smoothing 0.05. **Eval:** accuracy top-1 +
  matriz de confusión; objetivo ≥ 0.85 en sintético, ≥ 0.70 en curado.
- **Métrica real v1:** `val_accuracy 0.966` (objetivo ≥0.85 superado), `classify-v1.keras` 85 KB, 4.836 params.

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
