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

- **Req:** `{colors: ["#FF5733", ...], topK?: 1..10 = 5, harmony?: filtro opcional}`.
- **Resp 200:** `{palettes: [{type, colors: [5×Color], score}], fallback: bool}` rankeadas
  por `score` desc. `harmony` (si se envía) filtra/rerankea, ya no limita a una sola paleta.
- **Errores:** `400` HEX inválido (mismo texto que hoy), `422` validación pydantic.

## POST /api/v1/palette/identify (E→C)

- **Req:** `multipart/form-data {file: imagen}`.
- **Resp 200:** `{type, confidence: 0..1, colors: [10×Color con weight], fallback: bool}`.
- **Errores:** `400` imagen inválida, `503` idem extract.

## GET /api/v1/harmony · GET /api/v1/colors/roles · GET /health

Sin cambios respecto a `main` (listan `HarmonyType` / `ColorRole`, readiness).

## POST /api/v1/feedback (cierra el loop de entrenamiento)

- **Req:** `{imageSha256?, palette: {type, colors}, accepted: bool, score?: 0..1}`.
- **Resp 200:** `{ok: true}`. Persiste en `feedback` para re-entrenar G/C.
