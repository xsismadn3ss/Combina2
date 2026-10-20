export function getApiBaseUrl(): string {
  let raw = process.env.NEXT_PUBLIC_API_URL?.trim() || "http://localhost:8000";
  // Docker Compose: el contenedor web ve al backend como http://backend:8000,
  // pero el navegador necesita http://localhost:8000. Reescribe segun contexto
  // para que un unico NEXT_PUBLIC_API_URL funcione en ambos.
  if (typeof window === "undefined") {
    // SSR dentro de Docker: localhost -> backend
    if (raw.includes("localhost:8000")) raw = raw.replace("localhost", "backend");
    else if (raw.includes("127.0.0.1:8000")) raw = raw.replace("127.0.0.1", "backend");
  } else {
    // Browser: backend -> localhost
    if (raw.includes("backend:8000")) raw = raw.replace("backend", "localhost");
  }
  // remove trailing slash
  return raw.replace(/\/$/, "");
}

export function apiUrl(path: string): string {
  const base = getApiBaseUrl();
  // Normalize: if base already ends with /api/v1, don't duplicate
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (base.endsWith("/api/v1")) {
    // path is expected to start with /api/v1 -> strip duplicate
    if (normalizedPath.startsWith("/api/v1")) return `${base}${normalizedPath.slice(7)}`;
    return `${base}${normalizedPath}`;
  }
  return `${base}${normalizedPath}`;
}
