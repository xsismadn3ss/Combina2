"use client";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Loader2, ScanSearch } from "lucide-react";
import { UploadImage } from "@/components/UploadImage";
import { ResultCard } from "@/components/ResultCard";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, identifyPalette } from "@/lib/api/client";
import type { IdentifyResponse } from "@/lib/api/types";

type FlowState = "idle" | "uploading" | "done" | "error";

export default function IdentificarPage() {
  const [flow, setFlow] = useState<FlowState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<IdentifyResponse | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFileSelected = async (f: File) => {
    setFile(f);
    setResult(null);
    setFlow("uploading");
    try {
      const res = await identifyPalette(f);
      setResult(res);
      setFlow("done");
      toast.success("Ya encontramos la armonía de tu foto.");
    } catch (e) {
      toast.error(e instanceof ApiError && e.status === 400 ? "No pudimos procesar la imagen. Prueba con otra foto." : "No pudimos identificar los colores. Inténtalo de nuevo.");
      setFlow("error");
    }
  };

  const handleClear = () => {
    setFile(null);
    setResult(null);
    setFlow("idle");
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-2">
        <h1 className="flex items-center gap-2 font-heading text-2xl font-semibold">
          <ScanSearch className="size-6" /> Identificar paleta
        </h1>
        <p className="text-sm text-muted-foreground">
          Sube una foto para descubrir los colores y la armonía que la representan.
        </p>
      </div>

      <UploadImage
        onFileSelected={handleFileSelected}
        isLoading={flow === "uploading"}
        previewUrl={previewUrl}
        onClear={file ? handleClear : undefined}
      />

      {flow === "uploading" && (
        <Card>
          <CardContent className="space-y-4 pt-6">
            <div className="flex items-center gap-2 text-sm">
              <Loader2 className="size-4 animate-spin" /> Descubriendo la armonía de tu foto...
            </div>
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-2 w-full" />
            <div className="grid grid-cols-5 gap-2 sm:grid-cols-10">
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} className="h-20 w-full rounded-lg" />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {result && flow === "done" && <ResultCard result={result} />}

      {flow === "error" && (
        <Card>
          <CardContent className="pt-6 text-sm text-destructive">
            No pudimos identificar los colores. Prueba con otra foto.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
