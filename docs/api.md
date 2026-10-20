# API — contratos REST

Prefijo: `/api/v1`. Port de `GET /health`, `GET /harmony`, `GET /colors/roles` sin cambios
de forma. DTO base `Color{value: "#RRGGBB", role}` + `PaletteResponse{type, colors}` se
amplía con `weight?`, `score?`, `confidence?`, `fallback?`.

## POST /api/v1/palette/extract (red E)

- **Req:** `multipart/form-data {file: imagen}`.
- **Resp 200:** `{colors: [{value, role: "predominant", weight}], fallback: bool}` — 10 colores.
- **Errores:** `400` vacía/inválida (mismo mensaje que hoy), `503` modelo no cargado y sin
  fallback disponible.

## POST /api/v1/palette/generate (red G)

- **Req:** `{colors: ["#FF5733", ...] (1..5), topK?: 1..10 = 5, harmony?: filtro opcional}`.
  Si se envían más de 5 colores base, responde `400`.
- **Resp 200:** `{palettes: [{type, colors: [5×Color], score}], fallback: bool}` rankeadas
  por `score` desc. `harmony` (si se envía) filtra/rerankea, ya no limita a una sola paleta.
- **Errores:** `400` HEX inválido (mismo texto que hoy), `422` validación pydantic.

## POST /api/v1/palette/identify (E→C)

- **Req:** `multipart/form-data {file: imagen}`.
- **Resp 200:** `{type, confidence: 0..1, colors: [10×Color con weight], fallback: bool,
  confidenceCalibrated?: bool, needsRetry?: bool, suggestion?: string | null}`.
  Los tres últimos son aditivos y opcionales en clientes antiguos (camelCase, como
  `imageSha256` en `/feedback`).
- **Errores:** `400` imagen inválida, `503` idem extract.

### Confianza calibrada y aviso de reintento (B2)

- `confidenceCalibrated: true` indica que `confidence` es la probabilidad
  **calibrada** con la temperatura activa del manifiesto (`classify-v2`:
  T ajustada en el split de validación, no en test). Si es `false` (p. ej.
  rollback a `classify-v1`, motivo `legacy_40d`), `confidence` es la probabilidad
  cruda del softmax. `type` sigue siendo el argmax: la temperatura no lo cambia.
- `needsRetry: true` cuando **no** es fallback y se cumple alguna condición:
  - `confidence < IDENTIFY_MIN_CONFIDENCE` (env, default **0.70**), o
  - la imagen es casi monocroma / de baja información: el croma máximo
    (`hypot(a,b)` CIELAB) de los colores extraídos es **< 12** **y** además
    falta variedad de tono (spread de hues **< 20°** o **≤ 1** cluster de
    tono), aunque la confianza calibrada sea alta. Un dúo apagado con hues
    distintos (p. ej. `#E6DCD2`/`#D2E1EB`, spread alto) **no** dispara el aviso.
  Con `fallback: true` (predominancia clásica) siempre es `false`, porque el
  fallback no intenta clasificar armonía.
- `suggestion`: texto breve y accionable en español cuando `needsRetry` es `true`;
  `null` en cualquier otro caso. Ejemplo: "La foto no tiene suficiente variedad de
  color; prueba con mejor luz o una imagen más colorida."

**Justificación del umbral 0.70 (risk-coverage en test 80, con la T realmente
aplicada `T=0.266489`, ajustada en validación):**

| Umbral | Cobertura | Accuracy entre aceptadas |
|---|---|---|
| 0.50 | 100.00% | 97.50% |
| 0.60 | 100.00% | 97.50% |
| **0.70** | **100.00%** | **97.50%** |
| 0.80 | 98.75% | 97.47% |
| 0.90 | 97.50% | 97.44% |

Con la T aplicada, 0.70 acepta el 100% del test sin perder accuracy (97.5%);
subir a 0.80 o 0.90 solo rechaza fotos sin mejorar el acierto. El umbral actúa
como red de seguridad para fotos genuinamente dudosas, mientras que las
imágenes grises/monocromas se marcan por separado con el guard de baja
información (croma máximo < 12 **y** poca variedad de tono), independientemente
de la confianza calibrada.
La tabla se reproduce con
`python scripts/calibrate.py --slot classify --dry-run` (no ajusta nada:
usa la T guardada y el split de test como referencia).

## GET /api/v1/models (registro de versiones, B3)

- **Read-only.** No carga TensorFlow ni los `.keras`: solo lee
  `backend/models/manifest.json`. Seguro sin TF.
- **Resp 200:** objeto por slot (`extract`, `generate`, `classify`):

```json
{
  "classify": {
    "active": "classify-v2.keras",
    "previous": "classify-v1.keras",
    "versions": ["classify-v1.keras", "classify-v2.keras"],
    "files": {
      "classify-v2.keras": {
        "sha256": "<hex completo>",
        "createdAt": "2026-10-20T15:43:04Z",
        "gitSha": "bd0f5d0",
        "trainingRunId": 17,
        "tfVersion": "2.21.0",
        "kerasVersion": "3.15.1",
        "metrics": {"accuracy": 0.975},
        "calibration": {"method": "temperature", "calibrated": true, "temperature": 0.266489}
      }
    }
  }
}
```

- Solo expone nombres de archivo y metadatos; **nunca rutas absolutas**.
- `previous` puede ser `null`; `files` puede estar vacío en manifiestos legacy.

## GET /api/v1/harmony · GET /api/v1/colors/roles · GET /health

Sin cambios respecto a `main` (listan `HarmonyType` / `ColorRole`, readiness).

## POST /api/v1/feedback (cierra el loop de entrenamiento)

- **Req:** `{imageSha256?, palette: {type, colors}, accepted: bool, score?: 0..1}`.
- **Resp 200:** `{ok: true}`. Persiste en `feedback` para re-entrenar G/C.
