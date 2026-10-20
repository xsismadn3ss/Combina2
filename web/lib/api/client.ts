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
  const res = await fetch(apiUrl("/api/v1/harmony/"), { cache: "no-store" });
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
