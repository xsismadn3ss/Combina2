"use client";
import type { IdentifyResponse } from "@/lib/api/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Props = {
  result: IdentifyResponse;
};

export function ResultCard({ result }: Props) {
  const pct = Math.round(result.confidence * 100);
  const harmonyNames: Record<string, string> = { predominant: "Colores de la foto", complementario: "Colores complementarios", triada: "Colores en contraste", analogo: "Colores cercanos", monocromatica: "Tonos de un color" };
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2">
          <span>{harmonyNames[result.type.toLowerCase()] ?? "Combinación de colores"}</span>
          <Badge variant="secondary">Coincidencia: {pct}%</Badge>
        </CardTitle>
        <CardDescription>
          Esta combinación refleja los tonos que mejor representan tu foto.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {result.needsRetry ? (
          <div
            role="status"
            className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-200"
          >
            {result.suggestion ??
              "La foto no tiene suficiente variedad de color; prueba con mejor luz o una imagen más colorida."}
          </div>
        ) : null}
        <div className="grid grid-cols-5 gap-2 sm:grid-cols-10">
          {result.colors.map((c) => (
            <div
              key={c.value}
              className="flex flex-col overflow-hidden rounded-lg border"
              title="Color de la combinación"
            >
              <div className="h-12 w-full" style={{ backgroundColor: c.value }} />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
