"use client";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Palette, Loader2, Wand2 } from "lucide-react";
import { UploadImage } from "@/components/UploadImage";
import { Top10Grid } from "@/components/Top10Grid";
import { PaletteCarousel } from "@/components/PaletteCarousel";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { ApiError, extractPalette, generatePalettes, getHarmony } from "@/lib/api/client";
import type { ExtractResponse, GenerateResponse, HarmonyType } from "@/lib/api/types";
import { useTopK } from "@/hooks/useTopK";

type FlowState = "idle" | "uploading" | "selecting" | "generating" | "done" | "error";

export default function GenerarPage() {
  const { topK, setTopK } = useTopK();
  const [flow, setFlow] = useState<FlowState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [extract, setExtract] = useState<ExtractResponse | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [harmony, setHarmony] = useState<HarmonyType | "">("");
  const [harmonies, setHarmonies] = useState<string[]>([]);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  // cargar harmonias
  useEffect(() => {
    getHarmony()
      .then(setHarmonies)
      .catch(() => setHarmonies(["complementario", "triada", "analogo", "monocromatica"]));
  }, []);

  // preview URL lifecycle
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const reset = useCallback(() => {
    setFlow("idle");
    setFile(null);
    setExtract(null);
    setSelected([]);
    setResult(null);
  }, []);

  const handleFileSelected = async (f: File) => {
    setFile(f);
    setSelected([]);
    setResult(null);
    setFlow("uploading");
    try {
      const res = await extractPalette(f);
      setExtract(res);
      setFlow("selecting");
      toast.success("Colores extraídos");
    } catch (e) {
      toast.error(e instanceof ApiError && e.status === 400 ? "No pudimos procesar la imagen. Prueba con otra foto." : "No pudimos leer los colores de la imagen. Inténtalo de nuevo.");
      setFlow("error");
    }
  };

  const toggle = (hex: string) => {
    setSelected((prev) => (prev.includes(hex) ? prev.filter((h) => h !== hex) : [...prev, hex]));
  };

  const handleGenerate = async () => {
    if (selected.length === 0) {
      toast.error("Selecciona al menos un color");
      return;
    }
    setFlow("generating");
    try {
      const payload = {
        colors: selected,
        topK: topK,
        harmony: harmony ? (harmony as HarmonyType) : null,
      };
      const res = await generatePalettes(payload);
      setResult(res);
      setFlow("done");
      toast.success(`Generadas ${res.palettes.length} paletas`);
    } catch (e) {
      toast.error("No pudimos crear las paletas. Inténtalo de nuevo.");
      setFlow("error");
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-2">
        <h1 className="flex items-center gap-2 font-heading text-2xl font-semibold">
          <Palette className="size-6" /> Generar paleta
        </h1>
        <p className="text-sm text-muted-foreground">
          Sube una foto, elige los colores que te gustan y crea paletas para inspirarte.
        </p>
      </div>

      <UploadImage
        onFileSelected={handleFileSelected}
        isLoading={flow === "uploading" || flow === "generating"}
        previewUrl={previewUrl}
        onClear={reset}
      />

      {flow === "uploading" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Loader2 className="size-4 animate-spin" /> Extrayendo colores de la imagen
            </CardTitle>
            <CardDescription>Estamos descubriendo los colores de tu foto.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-5 gap-3">
            {Array.from({ length: 10 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full rounded-xl" />
            ))}
          </CardContent>
        </Card>
      )}

      {extract && (flow === "selecting" || flow === "generating" || flow === "done" || flow === "error") && (
        <Top10Grid colors={extract.colors} selected={selected} onToggle={toggle} />
      )}

      {(flow === "selecting" || flow === "generating" || flow === "done" || flow === "error") && extract && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Personaliza tus paletas</CardTitle>
            <CardDescription>Elige cuántas propuestas quieres y el estilo de combinación.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Cantidad de paletas</Label>
                 <Badge variant="outline">Cantidad de paletas: {topK}</Badge>
              </div>
              <Slider
                value={[topK]}
                min={1}
                max={10}
                step={1}
                onValueChange={(v) => {
                  const arr = Array.isArray(v) ? v : [v as number];
                  setTopK(arr[0] ?? 5);
                }}
                aria-label="Cantidad de paletas"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="harmony">Estilo de combinación</Label>
              <select
                id="harmony"
                value={harmony}
                onChange={(e) => setHarmony(e.target.value as HarmonyType | "")}
                className="flex h-9 w-full rounded-lg border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="">Cualquier estilo</option>
                {harmonies.map((h) => (
                  <option key={h} value={h}>{({ complementario: "Complementarios", complementarios: "Complementarios", triada: "En contraste", analogo: "Cercanos", analógica: "Cercanos", monocromatica: "Tonos de un color", monocromático: "Tonos de un color" } as Record<string, string>)[h.toLowerCase()] ?? h}</option>
                ))}
              </select>
            </div>
            <Button
              onClick={handleGenerate}
              disabled={selected.length === 0 || flow === "generating"}
              className="w-full gap-2"
            >
              {flow === "generating" ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              Crear paletas
            </Button>
            {flow === "error" && (
              <p className="text-sm text-destructive">Hubo un error. Revisa el toast o intenta de nuevo.</p>
            )}
          </CardContent>
        </Card>
      )}

      {flow === "generating" && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            <Loader2 className="mx-auto mb-2 size-6 animate-spin" /> Generando paletas...
          </CardContent>
        </Card>
      )}

      {result && flow === "done" && (
        <PaletteCarousel palettes={result.palettes} fallback={result.fallback} />
      )}
    </div>
  );
}
