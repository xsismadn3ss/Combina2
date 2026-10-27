# Modelos — E / G / C (TensorFlow/Keras, formato `.keras`)

Todo color de entrenamiento/inferencia se representa en **CIELAB normalizado**.
Conversiones y normalización viven en `backend/app/ml/color_spaces.py`.

## E — Extractora (`extract-v*.keras`; `extract-v1.keras` = 1 param)

- **Parámetros reales:** `extract-v1.keras` tiene **1 parámetro entrenable**: el
  camino desplegado es `HardKMeans` (KMeans determinista, sin pesos
  aprendidos); el único tensor es el `dummy` escalar de `HardKMeans` (para que
  el optimizador tenga una variable; no afecta a la salida). El MLP
  amortizado de `build_extract_model(use_kmeans=False)` sí tendría decenas de
  miles, pero no es la versión desplegada.
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
  600 train / 120 val / 24 test por defecto) + paletas de la DB de
  entrenamiento (si no alcanzan, relleno aleatorio offline determinista). Init de
  prototipos: MLP con identidad escalada + KMeans en features; reanimación
  de clusters muertos cada 4 épocas.

- **Hallazgo honesto:** red amortizada pura (MLP 3→128→128 + atención) con
  ~600 muestras **no alcanza ΔE <5** de forma estable (converge a ~8-12,
  cercano al límite de destilación reportado). Se entrega híbrido
  `HardKMeans` determinista que sí cumple **ΔE≈3.8 <5** manteniendo
  API/offline/determinista y sin `sklearn` en inferencia (KMeans 100% numpy,
  seed 42, paridad idéntica al maestro). **Si se exige amortizada pura sin
  KMeans en inferencia, umbral realista ~8-12.**

- **Eval de paridad:** ΔE medio **CIE76** vs `ClassicPaletteIdentifier` sobre
  set fijo de 10 mosaicos (size 64, `jpeg_quality=80`, seeds 5000..5009);
  objetivo ΔE < 5 — alcanzado **3.8119** medio y **14.1106** máximo con
  `extract-v1.keras` híbrido (fallback 0/10), por el camino real
  `inference.extract_image_colors` (`eval/run_eval.py`).

## G — Generadora (`generate-v1.keras` = 24.888 params)

- **Entrada:** (10, 3) paleta Lab + máscara multi-hot (10,) de seleccionados +
  embedding de armonía (4→8, opcional).
- **Arquitectura:** aplanado 10·3+10+8=48 → MLP 48→128→128 (ReLU, dropout 0.1) →
  cabeza paleta (15) + cabeza compatibilidad (1, sigmoide estilo O'Donovan).
- **Salida:** N paletas de 5 colores Lab + score (beam/muestreo con temperaturas
  distintas para diversidad); el servicio convierte a HEX y rankea.
- **Loss:** MSE-Lab + μ·distancia euclidiana Lab diferenciable (aproximación
  **CIE76**, no CIEDE2000; el nombre histórico `CIEDE2000 aprox` era incorrecto)
  + BCE del head de rating. La loss no cambia en B1.
- **Datos:** COLOURLovers + Color-Pedia + sintético HSL. **Eval (B1):** CIEDE2000
  **real** (Sharma et al.) contra ground truth; CIE76 se reporta como referencia
  histórica. Emparejamiento posicional de los 5 slots.
- **Línea base `generate-v1.keras`** (40 paletas seed 99, `eval/baseline.json`):
  CIEDE2000 medio **13.0981**, CIE76 medio **23.5952**, MSE **0.025599**.
  Score medio **0.9423** (std 0.0319, min 0.8088, max 0.9764). AUC
  score→buena reconstrucción (CIE76 por paleta ≤ 14) = **0.3516**:
  anti-correlacionado.
  `generate-v1.keras` 344 KB, 24.888 params.

- **Calibración (B2): NO aplicada.** El score está anti-correlacionado con la
  calidad (AUC 0.3516 < `calibration.require_auc_min = 0.55`), así que
  `scripts/calibrate.py` **rechaza** Platt y deja en el manifiesto
  `{method:"platt", calibrated:false, reason:"score_auc_below_threshold",
  auc:0.3516, a:null, b:null}`. No se invierte el score: eso cambiaría el
  ranking sin base. La inferencia solo aplicaría `apply_platt` si
  `calibrated:true` (hoy no ocurre).

## C — Clasificadora (`classify-v1.keras` = 4.836 params; `classify-v2.keras` = 5.732)

### v2 (activa) — pipeline-consistente

- **Entrada:** 54 dims de **conjunto** de colores, ordenadas por tono
  (`app/ml/color_features.palette_features`): 10 filas ×
  `[sin h, cos h, chroma/128, l/100, weight]` + resumen de estructura de tonos
  (4 dims). Es invariante al orden de E y entrena con la **misma**
  representación que usa la inferencia (`extract_image_colors` → `identify_image`).
- **Arquitectura:** MLP 54→64→32 (ReLU) → softmax 4
  (`complementario|triada|analogo|monocromatica`).
- **Loss:** cross-entropy + label smoothing 0.05.
- **Eval B1:** `pipeline` E→C sobre el split de test de **80 paletas**
  (seed 20042; train 600 seed 42, val 120 seed 10042). Los splits son
  **disjuntos**: las paletas de la DB de entrenamiento se reparten de forma
  **estratificada por `harmony_label`** (80/10/10 con semilla fija) en tres
  buckets, y el split de test usa siempre el bucket de test (índice 2), por lo
  que cualquier `test_seed` queda fuera de los buckets de train/val. Con al
  menos 3 paletas por clase cada split recibe una. El relleno sintético usa
  una semilla por split. Si la DB está vacía el resultado es idéntico al
  histórico (solo sintéticos).
  Tests: `tests/test_eval_gates.py` (incluye DB temporal poblada).
  Métricas: accuracy **0.975**, macro-F1
  **0.9757**, recall por clase {complementario 0.947, triada 0.957, analogo
  1.000, monocromatica 1.000}, confusión
  `[[18,0,0,1],[1,22,0,0],[0,0,16,0],[0,0,0,22]]`, NLL **0.1324**, ECE
  **0.0916** (20 bins).

- **Calibración (B2): temperatura ajustada SOLO en validación.** Split val de
  120 paletas (seed 10042, disjunto de test): ECE **0.1074 → 0.0210**, NLL
  **0.1377 → 0.0429**, T = **0.266489** (búsqueda determinista en `log-T`,
  rango 0.2–5.0). En test (referencia, la T NO se ajusta ahí): ECE
  **0.0916 → 0.0275**, NLL **0.1324 → 0.0929**, accuracy 0.975 invariante.
  El manifiesto guarda `{method:"temperature", calibrated:true, temperature,
  fittedOn:"pipeline-val", n:120, eceBefore/After, nllBefore/After,
  range:[0.2,5.0]}`. `inference._classify_with_model` aplica
  `softmax(log(p)/T)` (argmax invariante) y `IdentifyResponse.confidence`
  pasa a ser la probabilidad calibrada.

### v1 (rollback)

- 40 dims legacy (10·Lab normalizado + 10 pesos, dependiente del orden).
  Conservada en `versions`/`previous` sin cambios de comportamiento. En el
  mismo test pipeline de 80: accuracy **0.375**, macro-F1 **0.3519**.
  No se calibra (`{method:"temperature", calibrated:false, reason:"legacy_40d"}`):
  su dominio de features no es el de producción y el rollback debe mantener la
  semántica cruda hasta que exista una versión calibrada propia.

## Registro, versiones y rollback

`backend/models/manifest.json` (schemaVersion 2) es la única fuente de verdad y
se gestiona con `backend/app/services/model_registry.py` (Python puro, sin
TF/sklearn) y `scripts/manage_models.py`. Nunca se edita a mano.

Por slot:

- `active`: archivo que sirve la inferencia.
- `previous`: versión a la que vuelve `rollback` (intercambio reversible).
- `versions`: todas las versiones registradas. **Nunca se borran**.
- `files`: metadatos por versión — `sha256`, `createdAt`, `gitSha`,
  `trainingRunId`, `tfVersion`, `kerasVersion`, `metrics` (las que verifica
  `python -m eval.run_eval --check-manifest`, tolerancia 10%) y `calibration`
  (método, `calibrated`, parámetros y motivo de rechazo). Puede incluir
  `params` y `note`.

Resolución en inferencia (`inference.active_keras_path` →
`model_registry.resolve`):

1. Usa `active` si el archivo existe y su sha256 coincide.
2. Si falta o el hash no coincide, intenta `previous` (con su hash) y deja un
   warning en logs.
3. Si tampoco, devuelve `None` y la API responde por el fallback clásico con
   `fallback: true` (nunca se cae por un modelo ausente).

Los manifiestos legacy sin `files`/`sha256` se toleran: el archivo se usa sin
verificación (`verify` lo reporta como `unhashed`).

**Caché:** `inference.load_keras_model` indexa por `slot:archivo`, de modo que
un rollback en caliente carga la versión nueva en la siguiente petición. En
producción se recomienda reiniciar el proceso o llamar a
`inference.clear_model_cache()` tras promover/rollback.

La caché de features de evaluación (`.eval-cache/`, gitignored) incluye en su
clave la **versión y los 8 primeros hex del sha256 de la E resuelta**
(`extract_cache_tag`) y un **hash del contenido exacto del split**
(`split_content_hash`: HEX canónicos en orden + `harmony_label`). Reentrenar o
activar otra E, o cambiar las paletas de la DB de entrenamiento, invalida automáticamente
las features guardadas en lugar de reutilizar las de otro contenido.

```json
{
  "schemaVersion": 2,
  "updatedAt": "<ISO-8601>",
  "extract":  { "active": "extract-v1.keras",  "versions": ["extract-v1.keras"],
                "files": { "extract-v1.keras": { "sha256": "…", "trainingRunId": 14,
                                                "metrics": { "delta_e76_mean": 3.8119 } } } },
  "generate": { "active": "generate-v1.keras", "versions": ["generate-v1.keras"],
                "files": { "generate-v1.keras": { "sha256": "…", "trainingRunId": 15,
                  "calibration": { "method": "platt", "calibrated": false,
                                   "reason": "score_auc_below_threshold", "auc": 0.3516 } } } },
  "classify": { "active": "classify-v2.keras", "previous": "classify-v1.keras",
                "versions": ["classify-v1.keras", "classify-v2.keras"],
                "files": { "classify-v2.keras": {
                  "sha256": "…", "trainingRunId": 17,
                  "calibration": { "method": "temperature", "calibrated": true,
                                   "temperature": 0.266489, "fittedOn": "pipeline-val" } } } }
}
```

### Estado real de las versiones

- **C `classify-v2.keras` activa** (54 dims, pipeline-consistente, temperatura
  calibrada en validación). `classify-v1.keras` se conserva como `previous`
  (40 dims legacy, `calibrated:false`, `reason:"legacy_40d"`) para rollback.
- **G `generate-v1.keras` sin calibrar**: AUC score→reconstrucción 0.3516
  < 0.55, así que Platt no se aplica.
- E `extract-v1.keras` activa y sin calibración (`method:"none"`).

Cada `train_*.py` registra la nueva versión con
`model_registry.register_version` (nunca sobrescribe sin `--force`) e inserta
una fila en `training_runs` con `git_sha`, hiperparámetros, métricas y
versiones TF/Keras. Comandos y flujo completo en
[`retraining.md`](./retraining.md).

## Evaluación y gates (B1 + calibración B2)

- **Fuente única de umbrales:** `backend/eval/thresholds.json`.
- **CLI:** `python -m eval.run_eval [--strict] [--check-manifest] [--json r.json]
  [--md r.md] [--slot extract|generate|classify]` desde `backend/`.
- **Línea base medida:** `backend/eval/baseline.json` (números + versiones +
  sha256 de los modelos activos).
- Gates actuales:

| Slot | Métrica | Umbral |
|---|---|---|
| extract | ΔE76 medio | ≤ 5.0 |
| extract | ΔE76 máximo | ≤ 20.0 |
| generate | CIEDE2000 medio | ≤ 14.5 |
| generate | MSE | ≤ 0.05 |
| classify | accuracy (pipeline test) | ≥ 0.90 |
| classify | macro-F1 | ≥ 0.90 |
| classify | recall por clase | ≥ 0.85 |
| classify | ECE calibrada (solo si `calibrated:true`) | ≤ 0.10 |
| classify | NLL calibrada (solo si `calibrated:true`) | ≤ NLL cruda |

- **Calibración (B2):** `python scripts/calibrate.py --slot classify|generate|all
  [--dry-run] [--source auto|feedback|reconstruction]` desde `backend/` con
  `.venv-tf`. C ajusta T en validación y la guarda; G rechaza Platt si
  AUC < 0.55 (`eval/thresholds.json → calibration.require_auc_min`). El reporte
  de eval muestra params aplicados, ECE/NLL antes/después y, para G rechazada,
  `not-applied` sin fallar gates.

- **CIE76 vs CIEDE2000:** `losses.mean_delta_e` y la paridad de E usan CIE76
  (euclidiana en Lab). G se evalúa con **CIEDE2000 real**
  (`app/ml/metrics.py`, validado contra Sharma et al. 2005); su loss sigue
  usando la aproximación euclidiana diferenciable. No confundir ambos números
  (p.ej. G: CIE76 23.5952 vs CIEDE2000 13.0981).
- Sin TF, los slots ML se marcan `skipped`: exit 0 salvo `--strict`.
