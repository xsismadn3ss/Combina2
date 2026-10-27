# Next.js template

This is a Next.js template with shadcn/ui.

## Adding components

To add components to your app, run the following command:

```bash
npx shadcn@latest add button
```

This will place the ui components in the `components` directory.

## Using components

To use the components in your app, import them as follows:

```tsx
import { Button } from "@/components/ui/button";
```

## Export de feedback (admin)

La página `/admin/feedback` descarga el CSV a través del route handler
`/api/admin/feedback/export`. Antes consulta `/api/admin/feedback/count` con los
mismos filtros; si no hay filas, muestra un aviso y no genera un archivo vacío.
Las credenciales y el token se leen en el servidor Next (no son
`NEXT_PUBLIC_*`):

| Variable | Uso |
|---|---|
| `ADMIN_USERNAME` | Usuario de la pantalla de administración. |
| `ADMIN_PASSWORD` | Contraseña del admin; vacío ⇒ 503. |
| `FEEDBACK_EXPORT_TOKEN` | Token interno del backend; debe coincidir con el de `backend`. |
| `BACKEND_INTERNAL_URL` | URL privada del backend (`http://backend:8000` en compose). |

En local, estos valores vienen del `.env` de la raíz; en Railway se configuran
en el servicio `web` (y el token también en `backend`). El handler valida HTTP
Basic con `ADMIN_USERNAME`/`ADMIN_PASSWORD` y llama al backend con Bearer, así
que el token nunca llega al navegador.
