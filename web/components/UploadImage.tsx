"use client";
import { useCallback, useRef, useState } from "react";
import { Camera, ImagePlus, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { CameraCapture } from "@/components/CameraCapture";
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
  const [cameraOpen, setCameraOpen] = useState(false);

  const handleFile = useCallback(
    (file: File | null | undefined) => {
      if (!file) return;
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

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      handleFile(files[0]);
    },
    [handleFile]
  );

  // El atributo `capture` se aplica justo antes de abrir el picker para que en
  // móvil se abra la cámara nativa cuando el navegador no puede usar getUserMedia.
  const openFilePicker = useCallback((useNativeCamera: boolean) => {
    const input = inputRef.current;
    if (!input) return;
    if (useNativeCamera) {
      input.setAttribute("capture", "environment");
    } else {
      input.removeAttribute("capture");
    }
    input.click();
  }, []);

  const handleTakePhoto = useCallback(() => {
    const canUseBrowserCamera =
      typeof navigator !== "undefined" &&
      Boolean(navigator.mediaDevices?.getUserMedia) &&
      window.isSecureContext;

    if (!canUseBrowserCamera) {
      openFilePicker(true);
      return;
    }
    setCameraOpen(true);
  }, [openFilePicker]);

  const handleCaptured = useCallback(
    (file: File) => {
      handleFile(file);
    },
    [handleFile]
  );

  return (
    <Card>
      <CardContent className="pt-6">
        {validationError && (
          <p role="alert" className="mb-3 text-sm text-destructive">
            {validationError}
          </p>
        )}
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
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <Button
                type="button"
                variant="default"
                onClick={handleTakePhoto}
                disabled={isLoading}
                className="h-auto min-h-28 flex-col items-start gap-1 rounded-xl px-4 py-4 text-left whitespace-normal"
              >
                <span
                  aria-hidden
                  className="mb-1 flex size-10 items-center justify-center rounded-full bg-primary-foreground/15"
                >
                  <Camera className="size-5" />
                </span>
                <span className="text-sm font-semibold">Tomar foto</span>
                <span className="text-xs font-normal text-primary-foreground/80">
                  Usa la cámara ahora
                </span>
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => openFilePicker(false)}
                disabled={isLoading}
                className="h-auto min-h-28 flex-col items-start gap-1 rounded-xl px-4 py-4 text-left whitespace-normal"
              >
                <span
                  aria-hidden
                  className="mb-1 flex size-10 items-center justify-center rounded-full bg-muted"
                >
                  <ImagePlus className="size-5 text-muted-foreground" />
                </span>
                <span className="text-sm font-semibold">Elegir imagen</span>
                <span className="text-xs font-normal text-muted-foreground">
                  Desde tu galería o equipo
                </span>
              </Button>
            </div>

            <div
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
                "flex flex-col items-center justify-center gap-1 rounded-xl border border-dashed px-4 py-5 text-center transition-colors",
                dragOver ? "border-primary bg-primary/5" : "border-border bg-muted/30",
                isLoading && "pointer-events-none opacity-60"
              )}
            >
              <p className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Upload className="size-3.5" aria-hidden />
                También puedes arrastrar una imagen aquí
              </p>
              <p className="text-xs text-muted-foreground">
                PNG, JPG, WEBP, GIF o BMP · máximo 10 MB
              </p>
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif,image/bmp"
          className="hidden"
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
          aria-hidden
          tabIndex={-1}
        />
        <CameraCapture
          open={cameraOpen}
          onOpenChange={setCameraOpen}
          onCapture={handleCaptured}
          isLoading={isLoading}
        />
      </CardContent>
    </Card>
  );
}
