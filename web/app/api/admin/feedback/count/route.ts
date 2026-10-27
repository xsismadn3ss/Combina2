import {
  authorizeAdmin,
  callBackend,
  getUpstreamQuery,
  jsonResponse,
} from "@/lib/admin";

/**
 * Conteo previo a la descarga.
 *
 * La pantalla consulta este endpoint con los mismos filtros del export; si
 * devuelve 0, muestra un aviso y no genera un CSV vacío.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const COUNT_PATH = "/api/v1/feedback/count";

export async function GET(request: Request): Promise<Response> {
  const auth = authorizeAdmin(request);
  if (!auth.ok) return auth.response;

  let upstream: Response;
  try {
    upstream = await callBackend(
      COUNT_PATH,
      auth.config,
      getUpstreamQuery(request)
    );
  } catch {
    return jsonResponse(
      { detail: "No se pudo contactar con el backend para contar el feedback." },
      502
    );
  }

  if (upstream.status === 401 || upstream.status === 404) {
    return jsonResponse(
      {
        detail:
          "El backend rechazó el token de export. Revisa FEEDBACK_EXPORT_TOKEN en backend y web.",
      },
      503
    );
  }

  if (!upstream.ok) {
    const detail = await upstream.text();
    return new Response(detail, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") ?? "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  }

  const data = (await upstream.json()) as { count?: unknown };
  const count = typeof data.count === "number" ? data.count : 0;
  return jsonResponse({ count }, 200);
}
