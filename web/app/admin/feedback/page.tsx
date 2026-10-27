"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { ArrowLeft, Download, KeyRound, Loader2, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { ApiError, countFeedback, exportFeedbackCsv } from "@/lib/api/client";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

const ACCEPTED_ITEMS = [
  { label: "Todas", value: "all" },
  { label: "Solo aceptadas", value: "true" },
  { label: "Solo rechazadas", value: "false" },
] as const;

type AcceptedFilter = (typeof ACCEPTED_ITEMS)[number]["value"];

function exportErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Usuario o contraseña de administrador incorrectos.";
    if (error.status === 404) return "El export no está habilitado en el backend.";
    if (error.status === 502) return "No se pudo contactar con el backend.";
    if (error.status === 503) {
      return "El export no está disponible. Revisa ADMIN_USERNAME/ADMIN_PASSWORD y FEEDBACK_EXPORT_TOKEN en el servidor.";
    }
    return error.detail || `Error ${error.status}`;
  }
  return "No se pudo descargar el CSV. Inténtalo de nuevo.";
}

export default function AdminFeedbackPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [accepted, setAccepted] = useState<AcceptedFilter>("all");
  const [includeSha256, setIncludeSha256] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedUsername = username.trim();

    if (!trimmedUsername || !password) {
      setError("Introduce el usuario y la contraseña de administrador.");
      return;
    }

    if (from && to && from > to) {
      setError("La fecha «desde» no puede ser posterior a «hasta».");
      return;
    }

    setError(null);
    setIsDownloading(true);
    try {
      const filters = {
        username: trimmedUsername,
        password,
        from: from || undefined,
        to: to || undefined,
        accepted: accepted === "all" ? undefined : accepted === "true",
      };

      const count = await countFeedback(filters);
      if (count === 0) {
        toast.info("No hay feedback para los filtros seleccionados.");
        return;
      }

      const blob = await exportFeedbackCsv({
        ...filters,
        includeSha256,
      });

      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = `feedback-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(href), 1_000);

      toast.success(
        `CSV descargado con ${count} ${count === 1 ? "fila" : "filas"}`
      );
    } catch (caught) {
      const message = exportErrorMessage(caught);
      setError(message);
      toast.error(message);
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="flex items-center gap-2 font-heading text-2xl font-semibold">
          <ShieldCheck className="size-6" aria-hidden />
          Exportar feedback
        </h1>
        <p className="text-sm text-muted-foreground">
          Descarga el feedback recogido por la API como CSV UTF-8 para análisis
          o reentrenamiento. Esta pantalla es solo para administradores.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <KeyRound className="size-4" aria-hidden />
            Credenciales de administrador
          </CardTitle>
          <CardDescription>
            Se validan en el servidor de la aplicación y no se guardan en el
            navegador.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            onSubmit={handleSubmit}
            className="flex flex-col gap-5"
            aria-busy={isDownloading}
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex flex-col gap-2">
                <Label htmlFor="admin-username">Usuario</Label>
                <Input
                  id="admin-username"
                  name="username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="username"
                  spellCheck={false}
                  required
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="admin-password">Contraseña</Label>
                <Input
                  id="admin-password"
                  name="password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>

            <Separator />

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex flex-col gap-2">
                <Label htmlFor="feedback-from">Desde</Label>
                <Input
                  id="feedback-from"
                  type="date"
                  value={from}
                  onChange={(event) => setFrom(event.target.value)}
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="feedback-to">Hasta</Label>
                <Input
                  id="feedback-to"
                  type="date"
                  value={to}
                  onChange={(event) => setTo(event.target.value)}
                />
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="feedback-accepted">Respuestas</Label>
              <Select
                items={ACCEPTED_ITEMS}
                value={accepted}
                onValueChange={(value) =>
                  setAccepted(value as AcceptedFilter)
                }
              >
                <SelectTrigger id="feedback-accepted" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {ACCEPTED_ITEMS.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                className="size-4 accent-primary"
                checked={includeSha256}
                onChange={(event) => setIncludeSha256(event.target.checked)}
              />
              Incluir SHA-256 de la imagen
            </label>

            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}

            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" disabled={isDownloading}>
                {isDownloading ? (
                  <Loader2
                    data-icon="inline-start"
                    className="animate-spin"
                    aria-hidden
                  />
                ) : (
                  <Download data-icon="inline-start" aria-hidden />
                )}
                {isDownloading ? "Descargando…" : "Descargar CSV"}
              </Button>
              <Badge variant="outline">UTF-8 · CSV</Badge>
            </div>
          </form>
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">Qué incluye el CSV</CardTitle>
          <CardDescription>
            Encabezados estables:{" "}
            <code className="text-xs">
              id, created_at, accepted, score, palette_type, palette_json,
              colors_hex, image_sha256
            </code>
            .
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
          <p>
            El snapshot de la paleta y el estado de aceptación son suficientes
            para reentrenar. El SHA-256 es opcional y no se guardan imágenes
            originales.
          </p>
          <p>
            Los filtros de fecha usan UTC y el export omite las filas que no
            cumplan el rango.
          </p>
        </CardContent>
      </Card>

      <Link
        href="/config"
        className={buttonVariants({ variant: "ghost", size: "sm" })}
      >
        <ArrowLeft data-icon="inline-start" aria-hidden />
        Volver a preferencias
      </Link>
    </div>
  );
}
