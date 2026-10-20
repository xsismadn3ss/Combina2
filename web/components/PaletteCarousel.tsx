"use client";
import { useState } from "react";
import { ThumbsUp, ThumbsDown, Check } from "lucide-react";
import { toast } from "sonner";
import type { ScoredPalette } from "@/lib/api/types";
import { sendFeedback } from "@/lib/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ColorSwatch } from "@/components/ColorSwatch";

type Props = {
  palettes: ScoredPalette[];
  fallback?: boolean;
};

export function PaletteCarousel({ palettes, fallback }: Props) {
  const [feedbackDone, setFeedbackDone] = useState<Record<number, boolean>>({});

  const handleFeedback = async (index: number, palette: ScoredPalette, accepted: boolean) => {
    try {
      await sendFeedback({
        palette: { type: palette.type, colors: palette.colors, fallback: !!fallback },
        accepted,
        score: palette.score,
      });
      setFeedbackDone((prev) => ({ ...prev, [index]: true }));
      toast.success(accepted ? "¡Gracias! Feedback positivo enviado" : "Feedback negativo enviado");
    } catch {
      toast.error("No pudimos guardar tu opinión. Inténtalo de nuevo.");
    }
  };

  if (palettes.length === 0) {
    return <p className="text-sm text-muted-foreground">No encontramos paletas para esta selección. Prueba con otros colores.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <h3 className="font-medium">Tus propuestas de color</h3>
        <p className="text-xs text-muted-foreground">
          Toca un color para copiar su código HEX.
        </p>
      </div>

      {/* simple grid as carousel fallback - responsive */}
      <div className="grid gap-4 md:grid-cols-2">
        {palettes.map((p, idx) => (
          <Card key={`${p.type}-${idx}`} className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between text-sm">
                <span>Sugerencia {idx + 1} · {({ complementario: "Colores complementarios", triada: "Colores en contraste", analogo: "Colores cercanos", monocromatica: "Tonos de un color" } as Record<string, string>)[p.type.toLowerCase()] ?? "Combinación de colores"}</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-5 gap-1.5">
                {p.colors.map((c) => (
                  <ColorSwatch
                    key={`${c.value}-${c.role}`}
                    value={c.value}
                    copyable
                    swatchClassName="h-12 sm:h-16"
                  />
                ))}
              </div>

              {feedbackDone[idx] ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Check className="size-4" /> Gracias por compartir tu opinión
                </div>
              ) : (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleFeedback(idx, p, true)}
                    aria-label="Me gusta esta paleta"
                  >
                    <ThumbsUp className="size-4" /> Me gusta
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleFeedback(idx, p, false)}
                    aria-label="No me gusta esta paleta"
                  >
                    <ThumbsDown className="size-4" /> No me gusta
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
