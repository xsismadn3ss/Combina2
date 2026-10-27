import { createHash, timingSafeEqual } from "node:crypto";

/**
 * Autenticación de la administración del export.
 *
 * Vive solo en el servidor Next: las credenciales `ADMIN_USERNAME` /
 * `ADMIN_PASSWORD` y el `FEEDBACK_EXPORT_TOKEN` no deben importarse desde
 * componentes cliente.
 */

const ALLOWED_QUERY_KEYS = ["from", "to", "accepted", "include_sha256"] as const;

export type AdminConfig = {
  username: string;
  password: string;
  exportToken: string;
  backendUrl: string;
};

export type AdminAuthResult =
  | { ok: true; config: AdminConfig }
  | { ok: false; response: Response };

export function getAdminConfig(): AdminConfig {
  return {
    username: process.env.ADMIN_USERNAME?.trim() ?? "",
    password: process.env.ADMIN_PASSWORD ?? "",
    exportToken: process.env.FEEDBACK_EXPORT_TOKEN?.trim() ?? "",
    backendUrl: (
      process.env.BACKEND_INTERNAL_URL || "http://localhost:8000"
    ).replace(/\/$/, ""),
  };
}

function safeEqual(left: string, right: string): boolean {
  const leftHash = createHash("sha256").update(left, "utf8").digest();
  const rightHash = createHash("sha256").update(right, "utf8").digest();
  return timingSafeEqual(leftHash, rightHash);
}

function parseBasicAuth(
  authorization: string | null
): { username: string; password: string } | null {
  if (!authorization?.startsWith("Basic ")) return null;
  try {
    const decoded = Buffer.from(authorization.slice(6), "base64").toString("utf8");
    const separator = decoded.indexOf(":");
    if (separator < 0) return null;
    return {
      username: decoded.slice(0, separator),
      password: decoded.slice(separator + 1),
    };
  } catch {
    return null;
  }
}

export function jsonResponse(
  body: unknown,
  status: number,
  extraHeaders?: Record<string, string>
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  });
}

export function authorizeAdmin(request: Request): AdminAuthResult {
  const config = getAdminConfig();

  if (!config.username || !config.password) {
    return {
      ok: false,
      response: jsonResponse(
        {
          detail:
            "La administración no está configurada: faltan ADMIN_USERNAME/ADMIN_PASSWORD en el servidor web.",
        },
        503
      ),
    };
  }

  if (!config.exportToken) {
    return {
      ok: false,
      response: jsonResponse(
        {
          detail:
            "El export no está configurado: falta FEEDBACK_EXPORT_TOKEN en el servidor web.",
        },
        503
      ),
    };
  }

  const credentials = parseBasicAuth(request.headers.get("authorization"));
  if (
    !credentials ||
    !safeEqual(credentials.username, config.username) ||
    !safeEqual(credentials.password, config.password)
  ) {
    return {
      ok: false,
      response: jsonResponse(
        { detail: "Usuario o contraseña de administrador incorrectos." },
        401,
        { "WWW-Authenticate": 'Basic realm="Combina2 admin", charset="UTF-8"' }
      ),
    };
  }

  return { ok: true, config };
}

export function getUpstreamQuery(request: Request): string {
  const incoming = new URL(request.url);
  const upstream = new URLSearchParams();
  for (const key of ALLOWED_QUERY_KEYS) {
    const value = incoming.searchParams.get(key);
    if (value !== null) upstream.set(key, value);
  }
  return upstream.toString();
}

export async function callBackend(
  path: string,
  config: AdminConfig,
  query: string
): Promise<Response> {
  return fetch(`${config.backendUrl}${path}${query ? `?${query}` : ""}`, {
    headers: { Authorization: `Bearer ${config.exportToken}` },
    cache: "no-store",
  });
}
