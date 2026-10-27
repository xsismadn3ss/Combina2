# Demo E2E — Fase 6

Guía reproducible del flujo completo de Combina2: subir foto → extraer colores →
generar paletas → identificar una paleta → enviar feedback. La sección
[Resultados](#6-resultados) recoge la ejecución real del **2026-10-20** (rama
`22-combina2`), con los artefactos y capturas indicados en
[Evidencia](#5-evidencia).

Alcance: demo de producto (web + API). No sustituye a `eval.run_eval` ni a los
gates de CI.

> **Documentación histórica (Fase 6).** Este documento describe el estado de la
> rama `22-combina2` en 2026-10-20, cuando SQLite (`backend/training.db`) era la
> fuente de datos. Desde el issue #28 **PostgreSQL es la fuente de verdad**
> (`docs/data.md`): las referencias a `training.db` son evidencia histórica y no
> el flujo actual.

## 1. Prerrequisitos

- Repo en la rama de trabajo y `git status` revisado.
- Python 3.12 con el venv de entrenamiento `backend/.venv-tf` (TF 2.21 / Keras
  3.15). Es el que carga las redes E/G/C en la API; `backend/.venv` (3.14, sin
  TF) solo sirve para el camino fallback. Ver `backend/README.md`.
- Node 22 y dependencias de `web/` instaladas (`npm ci` o `npm install`).
- Modelos versionados en `backend/models/` y `manifest.json` consistente:

  ```powershell
  # desde backend/
  .venv-tf\Scripts\python.exe scripts\manage_models.py list
  .venv-tf\Scripts\python.exe scripts\manage_models.py verify
  ```

  `verify` debe salir con código 0. Si falla (archivo ausente o sha256
  distinto), **detén el demo**: no promociones ni ejecutes rollback para
  “arreglarlo” sin revisar `docs/retraining.md`.
- `backend/training.db` (histórico de la Fase 6): entonces se creaba sola y se
  versionaba como evidencia de los runs/feedback. **Desde #28 está deprecada**:
  PostgreSQL es la fuente de verdad y ningún código de `app/` la usa; el archivo
  se conserva en git sin modificar como evidencia histórica.
- Imágenes de prueba:
  - una foto con **variedad de color** (varias tonalidades, buena luz) para el
    camino feliz de `/identify`;
  - una foto **en escala de grises o casi monocroma** (una pared gris, una
    foto en blanco y negro o un cielo monocromo) para provocar `needsRetry`.
    El guard de colorfulness marca reintento cuando el croma máximo de los
    colores extraídos es < 12 **y** hay poca variedad de tono, **aunque la
    confianza calibrada sea alta**; no hace falta tocar
    `IDENTIFY_MIN_CONFIDENCE` (ver §2).

  Para que la reproducción sea determinista, las imágenes usadas en la
  evidencia se generan con:

  ```powershell
  # desde backend/ (sirve .venv o .venv-tf: solo usa Pillow)
  .venv-tf\Scripts\python.exe -c @"
  from PIL import Image
  # colorida.png: rejilla 3x2 de 240x240 con seis colores planos
  cols = [(230,57,70),(241,196,15),(29,53,87),
          (42,157,143),(231,111,81),(142,202,230)]
  img = Image.new('RGB', (240,240))
  for y in range(240):
      for x in range(240):
          img.putpixel((x,y), cols[(0 if y < 120 else 3) + x // 80])
  img.save('colorida.png')
  # gris.png: degradado horizontal 0..239 (monocromo)
  img = Image.new('RGB', (240,240))
  for y in range(240):
      for x in range(240):
          img.putpixel((x,y), (x,x,x))
  img.save('gris.png')
  "@
  ```

  Con `gris.png` la extracción devuelve 10 grises, el croma máximo es ~0 y el
  guard dispara `needsRetry` con confianza alta (ver §6).

## 2. Arranque

Backend (desde `backend/`):

```powershell
.venv-tf\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Sin variables, el umbral de confianza es `0.70` (`docs/api.md`). El caso de
reintento del demo se provoca con una **foto gris/monocroma**, no subiendo el
umbral. `IDENTIFY_MIN_CONFIDENCE` sigue siendo configurable, pero ya no es la
vía recomendada para reproducir el aviso: el guard de colorfulness lo activa
con una imagen de baja información y confianza alta.

Frontend (desde `web/`):

```powershell
npm run dev
```

- Web: <http://localhost:3000>
- API directa: <http://localhost:8000>
- Si la API no está en `localhost:8000`, define `NEXT_PUBLIC_API_URL` antes de
  `npm run dev` (por defecto el cliente usa `http://localhost:8000`, ver
  `web/lib/config.ts`).

Comprobación rápida:

```powershell
curl.exe -s http://localhost:8000/health
curl.exe -s http://localhost:8000/api/v1/models
```

`/health` debe responder `{"status":"ok"}` y `/api/v1/models` debe listar los
slots `extract`, `generate` y `classify` con su `active` no nulo. Guarda ambas
salidas como evidencia.

## 3. Flujo guiado (UI)

1. **Home (`/`)**: confirma que la página carga sin errores de consola. La home
   **no muestra el estado del backend**; esa comprobación se hace por API
   (`GET /health`, §2).
2. **`/generar` — subir foto**: arrastra o selecciona la imagen de prueba. Se
   muestra la preview y el skeleton mientras responde
   `POST /api/v1/palette/extract`.
3. **`/generar` — extracción**: aparece la tarjeta “Colores de tu foto” con los
   colores devueltos por la API (**hasta 10**; en la evidencia, 6). Selecciona
   **3** colores representativos; el contador muestra “3 colores elegidos”.
4. **`/generar` — generar**: ajusta el slider a **topK = 5**, deja el estilo en
   “Cualquier estilo” (o elige uno) y pulsa **Crear paletas**. Se muestran las
   propuestas con sus swatches (`POST /api/v1/palette/generate`). La UI **no
   muestra el score**: solo API/logs/feedback.
5. **`/generar` — feedback**: pulsa 👍 (“Me gusta”) en una paleta y 👎 (“No me
   gusta”) en otra. Cada acción llama a `POST /api/v1/feedback`; al confirmar,
   los botones se sustituyen por “Gracias por compartir tu opinión” y aparece
   el toast “¡Gracias! Feedback positivo enviado” / “Feedback negativo
   enviado”.
6. **`/identificar` — camino feliz**: sube la foto con variedad de color.
   Revisa el tipo de armonía, el badge de coincidencia y los swatches
   (`POST /api/v1/palette/identify`). La UI **no** muestra barra de confianza,
   HEX ni pesos: solo el badge “Coincidencia: N%” y los swatches. Anota
   `confidenceCalibrated` y `needsRetry` desde la respuesta de la API.
7. **`/identificar` — baja información**: usa la **foto gris/monocroma**. Debe
   observarse el aviso de reintento con el texto de `suggestion` **solo** si la
   respuesta no es `fallback`. Si el resultado es `fallback: true`,
   `needsRetry` siempre es `false` por diseño; documenta ese caso en vez de
   forzarlo.
8. **`/config` (opcional)**: mueve el slider de propuestas y recarga la página;
   el valor debe persistir (clave `localStorage: combina2:topK`).
9. **`/feedback` por API (opcional)**: si quieres evidencia en JSON, usa el
   comando de §4.5.

## 4. Comprobaciones por API (curl)

Ejecuta desde la raíz del repo. En PowerShell usa `curl.exe`.

### 4.1 Extract

```powershell
curl.exe -s -X POST http://localhost:8000/api/v1/palette/extract `
  -F "file=@ruta\a\foto-colorida.jpg"
```

Comprobar: `colors` con **hasta 10** entradas `{value, role, weight}` (la API
devuelve solo los tonos predominantes que encuentra; en la evidencia, 6),
`weight` suma 1 y `fallback` booleano. Imagen inválida → `400` con
`{"detail":"El archivo no es una imagen válida"}`.

### 4.2 Generate (topK 5)

Para evitar problemas de comillas entre PowerShell y `curl.exe`, envía el JSON
desde un archivo (`generate.json`):

```json
{
  "colors": ["#2A9D8F", "#1D3557", "#E63946"],
  "topK": 5,
  "harmony": null
}
```

```powershell
curl.exe -s -X POST http://localhost:8000/api/v1/palette/generate `
  -H "Content-Type: application/json" `
  --data-binary "@generate.json"
```

Sustituye los HEX de `generate.json` por tres colores devueltos en §4.1.
Comprobar: `palettes` rankeadas por `score` desc, cada una con 5 colores. Si se
envía `harmony`, actúa como filtro/rerank; si es `null`, el ranking puede
etiquetar las propuestas con cualquier armonía (en la evidencia API, `analogo`;
en la UI con “Cualquier estilo”, `complementario`). El `score` visible aquí no
se muestra en la UI.

### 4.3 Identify (camino feliz)

```powershell
curl.exe -s -X POST http://localhost:8000/api/v1/palette/identify `
  -F "file=@ruta\a\foto-colorida.jpg"
```

Comprobar: `type`, `confidence` en `[0,1]`, `fallback`,
`confidenceCalibrated`, `needsRetry`, `suggestion`. `colors` trae **hasta 10**
colores con `weight` (en la evidencia, 6).

### 4.4 Identify (baja información / needsRetry)

Usa `gris.png` de §1 (o cualquier foto en escala de grises / monocromo plano).
El guard de colorfulness debe disparar el aviso sin tocar el umbral:

```powershell
curl.exe -s -X POST http://localhost:8000/api/v1/palette/identify `
  -F "file=@ruta\a\gris.png"
```

Comprobar: `fallback: false`, `needsRetry: true` y `suggestion` con el texto de
imagen monocroma. Si la respuesta fuese `fallback: true`, documentarlo (el
fallback nunca pide reintento). El `confidence` puede ser alto; el aviso viene
del croma máximo < 12 y de la falta de variedad de tono, no de la confianza.

### 4.5 Feedback

`feedback.json` (copia una paleta realmente generada en §4.2; `imageSha256` es
opcional). `palette.fallback` es opcional y tolera `null`:

```json
{
  "palette": {
    "type": "analogo",
    "colors": [
      { "value": "#2A9D8F", "role": "primary" },
      { "value": "#1D3557", "role": "secondary" },
      { "value": "#E63946", "role": "secondary" },
      { "value": "#235C39", "role": "accent" },
      { "value": "#3B779B", "role": "accent" }
    ],
    "fallback": null
  },
  "accepted": true,
  "score": 0.8941118717193604
}
```

```powershell
curl.exe -s -X POST http://localhost:8000/api/v1/feedback `
  -H "Content-Type: application/json" `
  --data-binary "@feedback.json"
```

Comprobar: `{"ok":true}`. Casos validados por `backend/tests/test_contract.py`:

- `palette.fallback: false` → `200`.
- `palette.fallback: null` → `200` (se coerciona a `false`; antes del fix de
  #22 devolvía `422`).
- `palette` sin `fallback` → `200` (campo opcional).
- sin `accepted` → `422` (`accepted` sigue siendo obligatorio).

## 5. Evidencia

Artefactos de la ejecución, **versionados en [`docs/demo/evidence/`](./evidence/)**
(se generaron originalmente en `%TEMP%` y se copiaron al repo para que la
evidencia sea reproducible sin depender de una máquina):

- Entradas: `colorida.png` (rejilla 3×2, 240×240), `gris.png` (degradado
  horizontal, 240×240), `invalido.txt`.
- Respuestas JSON en `evidence/`: `health.json`, `models.json`, `extract.json`,
  `extract-invalid.json`, `generate-request.json`, `generate.json`,
  `identify-colorida.json`, `identify-gris.json`, `feedback-request.json`,
  `feedback.json`.
- Capturas en `evidence/`:
  - `01-generar-seleccion.png` — 6 colores extraídos, 3 seleccionados,
    “Cantidad de paletas: 5”, “Cualquier estilo”.
  - `02-generar-propuestas.png` — 5 propuestas (“Sugerencia 1..5”), anclas como
    primeros swatches, sin score visible.
  - `03-generar-feedback.png` — “Gracias por compartir tu opinión” tras 👍.
  - `04-identificar-colorida.png` — “Colores en contraste”, “Coincidencia:
    100%”, 6 swatches.
  - `05-identificar-needsretry.png` — “Tonos de un color”, “Coincidencia: 93%”,
    aviso de imagen monocroma.
  - `06-mobile-drawer.png` — drawer de navegación móvil abierto.
- Informe de evaluación de referencia: `evidence/eval-report.md` (12/12 gates,
  26 métricas de manifiesto OK). Los reportes locales `backend/eval-report.*`
  están gitignored y se regeneran con el comando de §5.
- CI verde de la rama `22-combina2` (run `39482806018`, commit `98a4b31`):
  `Backend runtime (3.14)`, `Backend ML (3.12 + TF 2.21, eval estricta)` y
  `Web (Node 22)` en success.

Comandos de reproducción de la evidencia API (desde la raíz, con el backend
levantado):

```powershell
$E = "$env:TEMP\opencode\fase6"
curl.exe -s http://localhost:8000/health                 | Out-File "$E\evidence\health.json"
curl.exe -s http://localhost:8000/api/v1/models          | Out-File "$E\evidence\models.json"
curl.exe -s -X POST http://localhost:8000/api/v1/palette/extract `
  -F "file=@$E\colorida.png"                             | Out-File "$E\evidence\extract.json"
curl.exe -s -X POST http://localhost:8000/api/v1/palette/extract `
  -F "file=@$E\invalido.txt"                             | Out-File "$E\evidence\extract-invalid.json"
curl.exe -s -X POST http://localhost:8000/api/v1/palette/generate `
  -H "Content-Type: application/json" `
  --data-binary "@$E\generate-request.json"              | Out-File "$E\evidence\generate.json"
curl.exe -s -X POST http://localhost:8000/api/v1/palette/identify `
  -F "file=@$E\colorida.png"                             | Out-File "$E\evidence\identify-colorida.json"
curl.exe -s -X POST http://localhost:8000/api/v1/palette/identify `
  -F "file=@$E\gris.png"                                 | Out-File "$E\evidence\identify-gris.json"
curl.exe -s -X POST http://localhost:8000/api/v1/feedback `
  -H "Content-Type: application/json" `
  --data-binary "@$E\feedback-request.json"              | Out-File "$E\evidence\feedback.json"
```

Verificaciones de cierre (desde `backend/` y `web/`):

```powershell
# backend/
.venv\Scripts\python.exe -m pytest tests/ -q
.venv-tf\Scripts\python.exe -m pytest tests/ -q
.venv-tf\Scripts\python.exe scripts\manage_models.py verify
.venv-tf\Scripts\python.exe scripts\manage_models.py rollback --slot classify --dry-run
.venv-tf\Scripts\python.exe -m eval.run_eval --strict --check-manifest `
  --json eval-report.json --md eval-report.md

# web/
npm run lint
npm run typecheck
npm run build
```

## 6. Resultados

Ejecución real del **2026-10-20** sobre `22-combina2` con TF 2.21 / Keras 3.15.
Todos los pasos: **PASS**.

| Paso | Comando/URL | Resultado observado | Evidencia |
|---|---|---|---|
| Health | `GET /health` | `200` · `{"status":"ok"}` | `evidence/health.json` |
| Registro de modelos | `GET /api/v1/models` | `200` · extract `extract-v1.keras` (previous `null`), generate `generate-v1.keras` (previous `null`), classify **`classify-v2.keras`** (previous `classify-v1.keras`, `temperature: 0.266489`, `calibrated: true`) | `evidence/models.json` |
| Extract | `POST /api/v1/palette/extract` (colorida) | `200` · 6 colores, `fallback:false`, pesos suman `1.0` | `evidence/extract.json` |
| Extract inválido | `POST /api/v1/palette/extract` (`invalido.txt`) | `400` · `{"detail":"El archivo no es una imagen válida"}` | `evidence/extract-invalid.json` |
| Generate (topK 5) | `POST /api/v1/palette/generate` (3 anclas, `harmony:null`) | `200` · 5 paletas × 5 colores, `fallback:false`, scores `0.8941 > 0.8493 > 0.8114 > 0.7866 > 0.7466`; anclas presentes en todas | `evidence/generate.json` |
| Identify feliz | `POST /api/v1/palette/identify` (colorida) | `200` · `type:"triada"`, `confidence:0.9999999`, `confidenceCalibrated:true`, `needsRetry:false`, `fallback:false`, `suggestion:null`, 6 colores | `evidence/identify-colorida.json` |
| Identify `needsRetry` | `POST /api/v1/palette/identify` (gris) | `200` · `type:"monocromatica"`, `confidence:0.9271`, `confidenceCalibrated:true`, `needsRetry:true`, `fallback:false`, `suggestion` = “La imagen es casi monocroma (gris o un solo tono); prueba con una foto con varios colores bien iluminada.”, 10 colores | `evidence/identify-gris.json` |
| Feedback | `POST /api/v1/feedback` (paleta con `fallback:false`) | `200` · `{"ok":true}` | `evidence/feedback.json` |
| Feedback tolerancia | `fallback:null` / campo ausente / sin `accepted` | `200` / `200` / `422` respectivamente (3 tests nuevos) | `backend/tests/test_contract.py` |
| Verify modelos | `manage_models.py verify` | `verify(extract, generate, classify): OK`, exit `0` | salida de consola |
| Rollback dry-run | `manage_models.py rollback --slot classify --dry-run` | `classify-v2.keras -> classify-v1.keras`, exit `0` (solo lectura) | salida de consola |
| Eval estricta | `eval.run_eval --strict --check-manifest` | **12/12 PASS**, exit `0`; extract ΔE76 3.8119/14.1106; generate CIEDE2000 13.0981, MSE 0.0256; classify accuracy 0.9750, macro-F1 0.9757, ECE 0.0275, NLL 0.0929; manifiesto 26/26 OK | `backend/eval-report.md`, `backend/eval-report.json` |
| Backend tests (sin TF) | `.venv\Scripts\python.exe -m pytest tests/ -q` | **185 passed, 9 skipped** | salida de consola |
| Backend tests (TF) | `.venv-tf\Scripts\python.exe -m pytest tests/ -q` | **194 passed** | salida de consola |
| Web CI local | `npm run lint` + `typecheck` + `build` | lint `0`, typecheck `0`, build `0` (Next.js 16.4.0) | salida de consola |

### UI observada

- **Home `/`**: carga con **0 errores de consola**. No muestra estado del
  backend (comportamiento actual; la comprobación es por API).
- **`/generar`**: 6 colores en “Colores de tu foto”, 3 seleccionados
  (“3 colores elegidos”), “Cantidad de paletas: 5”, 5 propuestas. Las anclas
  seleccionadas son los **primeros 3 swatches** de cada propuesta. Feedback 👍
  llama a `/feedback` (`200`) y muestra “Gracias por compartir tu opinión”.
  **No se muestra score**, ni HEX, ni pesos.
- **`/identificar` (colorida)**: “Colores en contraste” + badge “Coincidencia:
  100%” + 6 swatches. Sin barra de confianza, sin HEX/pesos.
- **`/identificar` (gris)**: “Tonos de un color” + badge “Coincidencia: 93%” +
  aviso de reintento con el texto de `suggestion`. Sin barra de confianza.
  La coincidencia del 93% no impide el aviso: lo dispara el guard de
  colorfulness.
- **Móvil**: drawer de navegación (“Navegación — Elige una sección para
  continuar”) abre y navega correctamente.

## 7. Notas e incidencias

- **Home sin estado del backend**: el guion anterior lo pedía, pero la UI no lo
  implementa. Se mantiene fuera del alcance de #22; la salud se verifica con
  `GET /health`.
- **Extract/identify devuelven hasta 10 colores**, no siempre 10. La evidencia
  colorida dio 6 y la gris 10.
- **Score solo en API/feedback**: la UI no lo pinta; `PaletteCarousel` lo
  reenvía en `POST /feedback`. El `generate.json` de API (harmony `analogo`)
  difiere del de UI (harmony `null` → “Colores complementarios”), por eso los
  tipos visibles en las capturas no coinciden con ese JSON.
- **Confianza solo como badge**: no hay barra de confianza ni HEX/pesos en
  `/identificar`; la respuesta API sí los incluye.
- **Fix de feedback (#22)**: `PaletteResponse.fallback` acepta `null` y lo
  coerciona a `false` (`field_validator` modo `before`), por lo que reenviar una
  paleta de `/generate` sin `fallback` o con `fallback:null` ya no da `422`.
  `accepted` sigue siendo obligatorio.
- **Artefactos fuera del repo**: capturas y JSON viven en
  `%TEMP%\opencode\fase6\`; solo los informes de eval y este documento se
  versionan. `backend/training.db` conserva las filas de feedback de la demo
  (evidencia histórica de Fase 6; desde #28 el feedback vive en PostgreSQL).
