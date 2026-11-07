"use client";
import { useCallback, useRef, useState } from "react";
import { Camera, ClipboardPaste, ImagePlus, Upload, X } from "lucide-react";
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

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      if (isLoading) return;
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            e.preventDefault();
            handleFile(file);
            return;
          }
        }
      }
      const files = e.clipboardData?.files;
      if (files && files.length > 0) {
        e.preventDefault();
        handleFile(files[0]);
      }
    },
    [handleFile, isLoading]
  );

  const handlePasteButton = useCallback(async () => {
    if (isLoading) return;
    try {
      if (!navigator.clipboard?.read) {
        setValidationError("Tu navegador no permite leer el portapapeles con un botón. Pega la imagen con Ctrl+V sobre esta zona.");
        return;
      }
      const items = await navigator.clipboard.read();
      for (const item of items) {
        const imageType = item.types.find((t) => t.startsWith("image/"));
        if (imageType) {
          const blob = await item.getType(imageType);
          const ext = imageType.split("/")[1] ?? "png";
          handleFile(new File([blob], `pegada-${Date.now()}.${ext}`, { type: imageType }));
          return;
        }
      }
      setValidationError("No encontramos una imagen en el portapapeles. Copia una imagen y vuelve a intentarlo.");
    } catch {
      setValidationError("No pudimos leer el portapapeles. Pega la imagen con Ctrl+V sobre esta zona.");
    }
  }, [handleFile, isLoading]);

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
    <Card onPaste={handlePaste}>
      <CardContent className="pt-6">
        {validationError && (
          <p role="alert" className="mb-3 text-sm text-destructive">
            {validationError}
          </p>
        )}
        {previewUrl ? (
          <div className="relative flex justify-center overflow-hidden rounded-xl border bg-muted/40">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Vista previa de la imagen cargada"
              className="mx-auto h-auto max-h-[60vh] w-full object-contain sm:max-h-[70vh]"
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
              onPaste={handlePaste}
              tabIndex={0}
              aria-label="Zona para pegar una imagen del portapapeles"
              className={cn(
                "flex flex-col items-center justify-center gap-1 rounded-xl border border-dashed px-4 py-5 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                dragOver ? "border-primary bg-primary/5" : "border-border bg-muted/30",
                isLoading && "pointer-events-none opacity-60"
              )}
            >
              <p className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Upload className="size-3.5" aria-hidden />
                Arrastra una imagen aquí o pégala con Ctrl+V
              </p>
              <p className="text-xs text-muted-foreground">
                PNG, JPG, WEBP, GIF o BMP · máximo 10 MB
              </p>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handlePasteButton}
                disabled={isLoading}
                className="mt-1 gap-1 text-xs"
              >
                <ClipboardPaste className="size-3.5" aria-hidden />
                Pegar desde portapapeles
              </Button>
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
