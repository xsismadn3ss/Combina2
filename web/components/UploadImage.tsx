"use client";
import { useCallback, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "cn";

type Props = {
  onFileSelected: (file: File) => void;
  isLoading?: boolean;
  previewUrl?: string | null;
  onClear?: () => void;
};

export function UploadImage({ onFileSelected, isLoading, previewUrl, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      setValidationError(null);
      if (!file.type.startsWith("image/")) {
        setValidationError("Elige un archivo de imagen en formato PNG, JPG, WEBP, GIF o BMP.");
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setValidationError("La imagen es demasiado grande. El tamaño máximo es 10 MB.");
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <Card>
      <CardContent className="pt-6">
        {validationError && <p role="alert" className="mb-3 text-sm text-destructive">{validationError}</p>}
        {previewUrl ? (
          <div className="relative overflow-hidden rounded-xl border">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Vista previa de la imagen cargada"
              className="h-64 w-full object-cover"
            />
            {onClear && (
              <Button
                variant="secondary"
                size="icon"
                aria-label="Quitar imagen"
                className="absolute right-2 top-2 size-8 rounded-full"
                onClick={onClear}
                disabled={isLoading}
              >
                <X className="size-4" />
              </Button>
            )}
          </div>
        ) : (
          <div
            role="button"
            tabIndex={0}
            aria-label="Zona para arrastrar y soltar imagen o hacer clic para seleccionar"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(e.dataTransfer.files);
            }}
            className={cn(
              "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              dragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:bg-muted/50",
              isLoading && "pointer-events-none opacity-60"
            )}
          >
            <span className="flex size-12 items-center justify-center rounded-full bg-muted">
              <Upload className="size-6 text-muted-foreground" aria-hidden />
            </span>
            <div className="space-y-1">
              <p className="text-sm font-medium">Arrastra una imagen o haz clic para seleccionar</p>
              <p className="text-xs text-muted-foreground">PNG, JPG, WEBP, GIF o BMP · máximo 10 MB</p>
            </div>
            <Button type="button" variant="outline" size="sm" disabled={isLoading}>
              Seleccionar imagen
            </Button>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif,image/bmp"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
          aria-hidden
          tabIndex={-1}
        />
      </CardContent>
    </Card>
  );
}
