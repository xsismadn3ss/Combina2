# Frontend — Next.js + shadcn (reemplaza `Web/` Blazor)

Decisión del dueño: reemplazo total, sin convivencia. En esta rama `Web/` ya fue
eliminado (borrón y cuenta nueva): `web/` parte de cero. El código anterior sigue
consultable en la rama `main` como referencia.

## Rutas (App Router)

- `/` — home: dos tarjetas (Generar / Identificar) + estado del backend (`/health`).
- `/generar` (historia 1) — `UploadImage` (foto desde la cámara del navegador o
  archivo/galería, más arrastrar y soltar) → skeleton → `Top10Grid` (selección multi) →
  `POST /generate` → `PaletteCarousel` (cards con score, HEX copiable) → 👍/👎 → `/feedback`.
- `/identificar` (historia 2) — `UploadImage` (foto desde la cámara del navegador o
  archivo/galería, más arrastrar y soltar) → `POST /identify` → `ResultCard`
  (tipo + barra de confianza + tira de 10 colores con pesos).

La cámara del navegador (`navigator.mediaDevices.getUserMedia`) solo está disponible en
contextos seguros: **HTTPS o `localhost`**. En HTTP por IP (por ejemplo, al abrir la app
desde el móvil contra el equipo de desarrollo) el formulario ofrece la cámara nativa del
dispositivo (`<input type="file" capture="environment">`) en lugar del modal web.
- `/config` — tema de la interfaz (Claro / Oscuro / Sistema) + URL del backend +
  `topK` por defecto (port de `Config.razor`).

## Componentes (shadcn)

`Card, Button, Badge, Skeleton, Toast (feedback), Tabs, Slider (topK), Progress (confianza),
Dialog (modal `CameraCapture` con preview y confirmación de la foto)`.
Cliente tipado: tipos TS generados del OpenAPI (`openapi-typescript`), `fetch` con
`NEXT_PUBLIC_API_URL`. Manejo de errores 400/422/503 con mensajes del `detail`.

## Skills de shadcn

El skill de shadcn para agentes vive en `.agents/skills/shadcn/` (project-scope de
OpenCode, instalado con `npx skills add shadcn/ui -s shadcn -a opencode --copy -y`).
`skills-lock.json` en la raíz fija la versión; para actualizarlo:

```bash
npx skills update shadcn -y
```

## Paridad con Blazor a verificar

`ImageForm/UploadImage`, `ColorPonderanceList`, `HarmonySelector`, `GenerateResult`,
estados de `GStateMachine` (idle→uploading→selecting→generating→done/error) y
`Identiy.razor`. Nada de la lógica de color vive en el front: solo presenta lo que
devuelve la API.
