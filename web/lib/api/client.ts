import { apiUrl } from "@/lib/config";
import type {
  CreatePaletteRequest,
  ExtractResponse,
  FeedbackRequest,
  FeedbackResponse,
  GenerateResponse,
  IdentifyResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    return (await res.json()) as T;
  }
  let detail = `Error ${res.status}`;
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") detail = data.detail;
    else if (Array.isArray(data?.detail)) {
      detail = data.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
    } else if (typeof data === "string") detail = data;
    else detail = JSON.stringify(data);
  } catch {
    try {
      const text = await res.text();
      if (text) detail = text;
    } catch {
      // ignore
    }
  }
  throw new ApiError(res.status, detail);
}

export async function getHealth(): Promise<{ status: string }> {
  const res = await fetch(apiUrl("/health"), { cache: "no-store" });
  // health is at /health not under /api/v1
  if (!res.ok) throw new ApiError(res.status, "Backend no disponible");
  return (await res.json()) as { status: string };
}

export async function getHarmony(): Promise<string[]> {
  const res = await fetch(apiUrl("/api/v1/harmony"), { cache: "no-store" });
  return handleResponse<string[]>(res);
}

export async function getColorRoles(): Promise<string[]> {
  const res = await fetch(apiUrl("/api/v1/colors/roles"), { cache: "no-store" });
  return handleResponse<string[]>(res);
}

export async function extractPalette(file: File): Promise<ExtractResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(apiUrl("/api/v1/palette/extract"), {
    method: "POST",
    body: fd,
  });
  return handleResponse<ExtractResponse>(res);
}

export async function generatePalettes(
  payload: CreatePaletteRequest
): Promise<GenerateResponse> {
  const res = await fetch(apiUrl("/api/v1/palette/generate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<GenerateResponse>(res);
}

export async function identifyPalette(file: File): Promise<IdentifyResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(apiUrl("/api/v1/palette/identify"), {
    method: "POST",
    body: fd,
  });
  return handleResponse<IdentifyResponse>(res);
}

export async function sendFeedback(
  payload: FeedbackRequest
): Promise<FeedbackResponse> {
  const res = await fetch(apiUrl("/api/v1/feedback"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<FeedbackResponse>(res);
}

export type AdminFeedbackFilters = {
  username: string;
  password: string;
  from?: string;
  to?: string;
  accepted?: boolean;
};

export type AdminFeedbackExportParams = AdminFeedbackFilters & {
  includeSha256?: boolean;
};

function encodeBasicCredentials(username: string, password: string): string {
  // Basic auth usa base64 de los bytes UTF-8, no de los code units UTF-16.
  const bytes = new TextEncoder().encode(`${username}:${password}`);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function buildFiltersQuery(params: AdminFeedbackFilters): URLSearchParams {
  const query = new URLSearchParams();
  if (params.from) query.set("from", params.from);
  if (params.to) query.set("to", params.to);
  if (typeof params.accepted === "boolean") {
    query.set("accepted", String(params.accepted));
  }
  return query;
}

async function apiErrorFromResponse(res: Response): Promise<ApiError> {
  const text = await res.text();
  let detail = `Error ${res.status}`;
  try {
    const data = JSON.parse(text) as { detail?: unknown };
    if (typeof data.detail === "string") detail = data.detail;
    else if (text) detail = text;
  } catch {
    if (text) detail = text;
  }
  return new ApiError(res.status, detail);
}

/**
 * Cuenta las filas que cumplirían el export antes de descargar.
 *
 * La UI lo usa para no generar un CSV vacío. No usa `apiUrl()` a propósito:
 * `/api/admin/*` vive en el servidor Next, no en el backend FastAPI.
 */
export async function countFeedback(
  filters: AdminFeedbackFilters
): Promise<number> {
  const qs = buildFiltersQuery(filters).toString();
  const res = await fetch(
    `/api/admin/feedback/count${qs ? `?${qs}` : ""}`,
    {
      headers: {
        Authorization: `Basic ${encodeBasicCredentials(filters.username, filters.password)}`,
      },
      cache: "no-store",
    }
  );

  if (!res.ok) throw await apiErrorFromResponse(res);

  const data = (await res.json()) as { count?: unknown };
  return typeof data.count === "number" ? data.count : 0;
}

/**
 * Descarga el CSV de feedback a través del proxy de administración de Next.
 *
 * No usa `apiUrl()` a propósito: `/api/admin/*` vive en el servidor Next y no
 * en el backend FastAPI. Las credenciales se validan allí y el token del
 * backend nunca llega al navegador.
 */
export async function exportFeedbackCsv(
  params: AdminFeedbackExportParams
): Promise<Blob> {
  const query = buildFiltersQuery(params);
  query.set("include_sha256", params.includeSha256 ? "true" : "false");

  const qs = query.toString();
  const res = await fetch(
    `/api/admin/feedback/export${qs ? `?${qs}` : ""}`,
    {
      headers: {
        Authorization: `Basic ${encodeBasicCredentials(params.username, params.password)}`,
      },
      cache: "no-store",
    }
  );

  if (!res.ok) throw await apiErrorFromResponse(res);

  return await res.blob();
}
