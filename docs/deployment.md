# Despliegue en Railway

Arquitectura de producción de Combina2: **dos servicios Docker** del mismo repo
y **un Postgres gestionado** de Railway para el feedback de runtime.

```
navegador
   │  https://<web-publica>/api/v1/...      (same-origin)
   ▼
[web] Next.js standalone (público, puerto 3000)
   │  rewrite a BACKEND_INTERNAL_URL
   │  http://backend.railway.internal:8000  (red privada, HTTP)
   ▼
[backend] FastAPI + TensorFlow (privado, puerto 8000)
   │  /app/models/*.keras + manifest.json   (solo artefactos)
   │  DATABASE_URL (pg8000)
   ▼
[Postgres] (solo feedback de runtime; datos de entrenamiento NO van a prod)
```

- **web** es el único servicio con dominio público:
  `https://combina2.up.railway.app`.
- **backend** no expone dominio público: solo se alcanza por la red privada de
  Railway (`http://<servicio>.railway.internal:<puerto>`, **sin `https`**).
- El frontend usa **same-origin**: en producción `NEXT_PUBLIC_API_URL` queda
  vacía, el navegador pide rutas relativas `/api/v1/...` y Next las proxya al
  backend. No hay CORS ni DNS privado expuesto al cliente.
- **Postgres** guarda únicamente el feedback (`feedback`). La inferencia no
  depende de la DB: si está caída, `/palette/*`, `/health` y `/ready` siguen
  funcionando; solo `POST /feedback` responde `503` y el export falla.

## Servicios y root directories

| Servicio | Root directory | Dockerfile | Config | Healthcheck |
|---|---|---|---|---|
| `backend` | `backend` | `backend/Dockerfile` | Service settings | `GET /ready` (120 s) |
| `web` | `web` | `web/Dockerfile` | Service settings | `GET /` (60 s) |
| `Postgres` | — | imagen gestionada | plugin de Railway | `pg_isready` |

En Railway, cada servicio se crea apuntando al repo y se configura el **root
directory** correspondiente; así el Dockerfile de cada carpeta se resuelve
correctamente. La configuración se guarda como **service settings** de Railway
(builder `DOCKERFILE`, `rootDirectory`, `dockerfilePath`, healthcheck y política
de reinicio), no en `railway.toml`.

> `railway.json` / `railway.toml` están **deprecados** por Railway. No se usan en
> este repo. Para infraestructura reproducible, la alternativa oficial es IaC en
> `.railway/railway.ts` (`railway config pull` / `plan` / `apply`).

## Requisitos de imagen

- Backend: **Python 3.12** + `tensorflow-cpu==2.21.0` y `keras==3.15.1` (extra
  `.[ml]` de `backend/pyproject.toml`), las mismas versiones con las que se
  entrenaron los `.keras`. `libgomp1` se instala por OpenMP.
- Frontend: Node 20, salida `standalone`.
- Los `.keras` y `manifest.json` viven en el repo (`backend/models/`) y se
  copian a la imagen.

## Variables

### Servicio `web`

| Variable | Valor | Notas |
|---|---|---|
| `BACKEND_INTERNAL_URL` | `http://backend.railway.internal:8000` | **Build arg y runtime**: `next.config.ts` compila los rewrites y el route handler `/api/admin/feedback/export` lo usa para proxyar el CSV. Railway inyecta las variables como build args si el Dockerfile las declara con `ARG` (ya está). Usa el nombre real del servicio backend. |
| `NEXT_PUBLIC_API_URL` | *(vacía / no definir)* | Vacía ⇒ same-origin (`/api/v1/...`). **No** poner `localhost`. |
| `NEXT_PUBLIC_SITE_URL` | `https://combina2.up.railway.app` | **Build-time**: base de `canonical`, `og:url`, `og:image` y `twitter:image`. Si cambia el dominio, redeployar `web`. |
| `ADMIN_USERNAME` | `admin` (o el que elijas) | Usuario de `/admin/feedback`. Se valida **solo en el servidor Next**; no usar `NEXT_PUBLIC_*`. |
| `ADMIN_PASSWORD` | secreto | Contraseña del admin. Vacío ⇒ administración deshabilitada (503). |
| `FEEDBACK_EXPORT_TOKEN` | **el mismo** que en `backend` | El route handler de Next lo usa para llamar al backend; nunca llega al navegador. |
| `PORT` | inyectada por Railway | Next standalone escucha en `$PORT`. |

### Servicio `backend`

| Variable | Valor | Notas |
|---|---|---|
| `PORT` | inyectada por Railway | El `CMD` expande `${PORT:-8000}`. |
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Plugin Postgres de Railway (`postgresql://...`). **El backend normaliza el esquema** a `postgresql+pg8000://` (`normalize_database_url`), también si llega como `postgres://`. Se usa pg8000 en runtime para evitar el segfault de psycopg binario con TensorFlow en Linux. **Obligatoria** para el feedback. |
| `FEEDBACK_EXPORT_TOKEN` | secreto largo | Bearer de `GET /api/v1/feedback/export.csv`. Si falta, el endpoint responde 404. |
| `IDENTIFY_MIN_CONFIDENCE` | opcional, default `0.70` | Umbral de `/identify` para `needsRetry`. |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Ya fijada en la imagen; reduce ruido de TF. |
| `TRAINING_DATABASE_URL` | **no definir** | Los datos de entrenamiento se quedan en local. |

Alta del plugin y variables (no ejecutar desde este repo; documentación):

```bash
railway add --database postgres --json
railway variable set 'DATABASE_URL=${{Postgres.DATABASE_URL}}' --service backend
railway variable set FEEDBACK_EXPORT_TOKEN='<secreto>' --service backend
# La web necesita el MISMO token para proxyar el CSV, más el admin:
railway variable set FEEDBACK_EXPORT_TOKEN='<secreto>' --service web
railway variable set ADMIN_USERNAME='admin' --service web
railway variable set ADMIN_PASSWORD='<secreto-admin>' --service web
```

### Interfaz de administración

La pantalla pública es `https://<dominio-web>/admin/feedback` (también
`http://localhost:3000/admin/feedback` en local). Pide `ADMIN_USERNAME` y
`ADMIN_PASSWORD`; el route handler de Next `/api/admin/feedback/export`:

1. valida las credenciales con comparación en tiempo constante,
2. llama al backend con `Authorization: Bearer $FEEDBACK_EXPORT_TOKEN`,
3. devuelve el CSV al navegador.

Así el token del backend **no se incluye en el bundle** ni se escribe en el
navegador. `ADMIN_USERNAME`/`ADMIN_PASSWORD` son variables de runtime del
servicio `web`; no uses `NEXT_PUBLIC_*`. Si `ADMIN_PASSWORD` está vacío, la
pantalla responde 503 hasta configurarlo.

**Pre-deploy command** del servicio `backend` (Service settings → Deploy):

```bash
alembic upgrade head
```

La imagen copia `alembic.ini` y `migrations/` a `/app`; `env.py` toma la URL de
`DATABASE_URL` y usa un advisory lock para que varias réplicas no migren a la
vez. El comando es idempotente.

## SEO y previews (Open Graph / Twitter)

El metadata vive en `web/app/layout.tsx` y usa la imagen
`web/public/combina2.jpg` (1408×768) como preview:

- `title`: **Combina2**
- `description`: *Genera o identifica paletas de colores tomando una imagen ✨*
- `openGraph.images` y `twitter.images` → `https://combina2.up.railway.app/combina2.jpg`
- `twitter:card`: `summary_large_image`
- `canonical` y `og:url` → `https://combina2.up.railway.app`

La imagen debe estar disponible sin autenticación (Next la sirve desde
`public/`). Para comprobar un preview después de un deploy:

```bash
curl -s -A "facebookexternalhit/1.1" https://combina2.up.railway.app/ \
  | grep -E 'og:(title|description|image)|twitter:(card|image)'
curl -sI https://combina2.up.railway.app/combina2.jpg   # 200 image/jpeg
```

Además se puede re-escanear en el [Sharing Debugger de Facebook](https://developers.facebook.com/tools/debug/),
el [Card Validator de X](https://cards-dev.twitter.com/validator) o el
[Post Inspector de LinkedIn](https://www.linkedin.com/post-inspector/). Si el
preview queda cacheado, fuerza un nuevo scrape en la herramienta.

> **La inferencia no depende de Postgres.** `backend/training.db` (SQLite) está
> deprecado y **no se despliega** (el `Dockerfile` solo copia `app/`, `models/`,
> `alembic.ini` y `migrations/`, y `.dockerignore` lo excluye). En producción
> solo se usa Postgres para `feedback`; `/ready` y las redes E/G/C dependen
> únicamente de `models/*.keras` + `models/manifest.json`. Si Postgres cae,
> `/health` y `/ready` siguen respondiendo y `POST /feedback` devuelve `503` con
> `Retry-After: 30`.

## Backups, restore y export

> `pg_dump`/`pg_restore` hablan libpq: usan `postgresql://...`, **no** el
> esquema `postgresql+pg8000://...` ni `postgresql+psycopg://...` de SQLAlchemy.
> Si copias la URL local de `.env`, quita el sufijo `+psycopg`; si vienes del
> backend, quita `+pg8000`:
>
> ```bash
> PG_URL="${DATABASE_URL/postgresql+psycopg:/postgresql:}"
> # también válido si la URL ya es postgresql://...
> ```

### Backup

```bash
# URL pública/privada del plugin (dashboard → Postgres → Connect), sin +psycopg.
pg_dump "$PG_URL" --no-owner --no-privileges -Fc -f combina2-feedback.dump
```

Railway también ofrece backups del plugin desde el dashboard; programa una
política acorde con la retención (indefinida por defecto).

### Restore

```bash
pg_restore --clean --if-exists --no-owner --no-privileges \
  -d "$PG_URL" combina2-feedback.dump
alembic upgrade head   # por si el dump era de un esquema anterior
```

### Export CSV (vía web público o proxy)

El servicio `web` ya proxya `/api/v1/*` al backend privado, así que el `curl`
puede ir contra la URL pública de `web`:

```bash
curl -fsS -H "Authorization: Bearer $FEEDBACK_EXPORT_TOKEN" \
  "https://combina2.up.railway.app/api/v1/feedback/export.csv" -o feedback.csv
```

Alternativa sin endpoint público: `railway connect Postgres` abre un proxy
local; usa esa URL en `DATABASE_URL` y ejecuta el script:

```bash
cd backend
DATABASE_URL="postgresql://postgres:...@127.0.0.1:<puerto>/railway" \
  python scripts/export_feedback_csv.py --output feedback.csv
```

Recuerda que `*.railway.internal` **no resuelve fuera** de Railway: fuera de
Railway hay que usar el proxy de `railway connect` o una URL pública.

## Build y despliegue

```bash
# desde la raíz del repo
railway link --project combina2
railway service link backend && railway up --service backend
railway service link web && railway up --service web
```

También se puede desplegar desde el dashboard (servicio → Deploy) o conectando
cada servicio al repo de GitHub con su `rootDirectory` (`backend` / `web`) para
auto-deploy en cada push. En ese caso, el build de `web` recibe
`BACKEND_INTERNAL_URL` como variable/build arg.

El build del backend instala TensorFlow (unos minutos la primera vez). El
frontend necesita `BACKEND_INTERNAL_URL` disponible **en el build**; si se
cambia el nombre del servicio backend, hay que redeployar `web` para recompilar
los rewrites.

### Logs

```bash
railway logs --service backend --lines 200
railway logs --service web --lines 200
railway logs --service backend --build --lines 200   # build/deploy
```

Si el deploy queda unhealthy, revisar primero el log de build (fallo de
instalación de TF) y luego el arranque (`uvicorn`, carga de modelos).

## Healthchecks

- **Backend `GET /ready`** (`backend/app/main.py` →
  `inference.readiness_report()`): 200 solo si los tres slots (`extract`,
  `generate`, `classify`) resuelven su `.keras` activo, el archivo existe y su
  `sha256` coincide, **y** el modelo carga en Keras. Si algo falla devuelve 503
  con `ready:false`, `fallback:true` y el `error` por slot. `storage` siempre es
  `models_only`. El campo `database:{configured,reachable}` es informativo
  (timeout corto + caché 10 s) y **no cambia** el 200/503.
- **Backend `GET /health`**: liveness puro, 200 `{"status":"ok"}` sin cargar
  modelos (no penaliza el arranque).
- **Web `GET /`**: 200 cuando Next sirve la home.
- La primera llamada a `/ready` carga los modelos en memoria; las siguientes
  usan la caché. El timeout de 120 s cubre el arranque en frío.

## Rollback

### Código

1. Railway → servicio → **Deployments**.
2. Elegir el deployment anterior sano → **Redeploy**.
3. Verificar `GET /ready` (vía web: `https://<web-publica>/ready`) y el smoke
   E2E.

El redeploy arranca un proceso nuevo: la caché de modelos se descarta.

### Modelos

El registro vive en `backend/models/manifest.json` (ver
`docs/retraining.md` y `backend/models/README.md`). Un rollback mueve
`active`/`previous`; **no borra versiones**.

```bash
cd backend
.venv\Scripts\python.exe scripts\manage_models.py list
.venv\Scripts\python.exe scripts\manage_models.py rollback --slot classify
.venv\Scripts\python.exe scripts\manage_models.py verify
```

Después:

1. Commit del `manifest.json` (y de los `.keras` si el cambio los incluye).
2. `railway up --service backend` o redeploy del mismo commit.
3. Confirmar en `/ready` que el slot reporta el archivo esperado y
   `loaded: true`.

> Nota: `manage_models.py` actúa **en local**; producción solo cambia cuando se
> despliega el manifiesto actualizado.

## Smoke E2E

Con la URL pública del servicio `web`:

```bash
WEB="https://combina2.up.railway.app"

curl -fsS "$WEB/health"     # {"status":"ok"}
curl -fsS "$WEB/ready"      # ready:true, 3 modelos loaded:true, storage:"models_only"

curl -fsS -X POST "$WEB/api/v1/palette/generate" \
  -H "Content-Type: application/json" \
  -d '{"colors":["#FF0000"],"harmony":"complementario","topK":3}'

curl -fsS -X POST "$WEB/api/v1/palette/extract" \
  -F "file=@ruta/a/foto.jpg"

curl -fsS -X POST "$WEB/api/v1/palette/identify" \
  -F "file=@ruta/a/foto.jpg"

curl -fsS -X POST "$WEB/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{"palette":{"type":"triada","colors":[{"value":"#FF0000","role":"primary"}]},"accepted":true,"score":0.9}'

curl -fsS -H "Authorization: Bearer $FEEDBACK_EXPORT_TOKEN" \
  "$WEB/api/v1/feedback/export.csv" | head -n 2
```

Comprobaciones esperadas:

- `/ready`: HTTP 200, `ready:true`, `fallback:false`, los tres modelos
  `loaded:true`, `storage:"models_only"`.
- `/palette/identify` y `/palette/extract`: `fallback:false` (modelos reales).
  `identify` incluye `needsRetry` y `confidenceCalibrated`.
- `POST /feedback`: 200 `{"ok":true}`; si Postgres está caído, 503 con
  `Retry-After: 30`. `export.csv` devuelve 200 con el token y 401 sin él.
- Si `fallback:true`, el backend no cargó TF/modelos: revisar `/ready` y los
  logs del servicio backend.
