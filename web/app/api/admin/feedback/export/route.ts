import {
  authorizeAdmin,
  callBackend,
  getUpstreamQuery,
  jsonResponse,
} from "@/lib/admin";

/**
 * Proxy server-side del export CSV.
 *
 * El navegador se autentica con las credenciales de administrador; este
 * handler las valida y llama al backend con el token interno.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const EXPORT_PATH = "/api/v1/feedback/export.csv";

export async function GET(request: Request): Promise<Response> {
  const auth = authorizeAdmin(request);
  if (!auth.ok) return auth.response;

  let upstream: Response;
  try {
    upstream = await callBackend(
      EXPORT_PATH,
      auth.config,
      getUpstreamQuery(request)
    );
  } catch {
    return jsonResponse(
      { detail: "No se pudo contactar con el backend para exportar el feedback." },
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

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="feedback.csv"',
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
