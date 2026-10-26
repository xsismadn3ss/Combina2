"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Camera, Check, Loader2, RefreshCw, SwitchCamera, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "cn";

type FacingMode = "environment" | "user";

type CameraStatus = "starting" | "live" | "captured" | "error";

type CameraErrorKind = "permission" | "not-found" | "in-use" | "insecure" | "unsupported" | "unknown";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCapture: (file: File) => void;
  isLoading?: boolean;
};

const MAX_CAPTURE_EDGE = 1600;
const JPEG_QUALITY = 0.85;

const ERROR_MESSAGES: Record<CameraErrorKind, string> = {
  permission:
    "No pudimos acceder a la cámara. Revisa los permisos del navegador para este sitio y vuelve a intentarlo.",
  "not-found":
    "No encontramos una cámara disponible en este dispositivo. Puedes usar la cámara nativa o elegir una imagen.",
  "in-use":
    "La cámara está siendo usada por otra aplicación. Cierra las apps que la estén usando e inténtalo de nuevo.",
  insecure:
    "La cámara del navegador requiere HTTPS o localhost. Puedes usar la cámara nativa del dispositivo.",
  unsupported:
    "Este navegador no permite abrir la cámara desde la página. Puedes usar la cámara nativa del dispositivo.",
  unknown:
    "No pudimos abrir la cámara. Prueba con la cámara del dispositivo o elige una imagen.",
};

const DESCRIPTIONS: Record<CameraStatus, string> = {
  starting: "Pidiendo permiso para usar la cámara…",
  live: "Apunta la cámara y pulsa el disparador.",
  captured: "Revisa la foto antes de usarla.",
  error: "Puedes abrir la cámara nativa del dispositivo o elegir una imagen.",
};

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function errorName(error: unknown): string {
  if (error instanceof DOMException) return error.name;
  if (typeof error === "object" && error !== null && "name" in error) {
    return String((error as { name?: unknown }).name);
  }
  return "";
}

function classifyError(error: unknown): CameraErrorKind {
  switch (errorName(error)) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return "permission";
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "not-found";
    case "NotReadableError":
    case "TrackStartError":
      return "in-use";
    default:
      return "unknown";
  }
}

async function requestCameraStream(facingMode: FacingMode): Promise<MediaStream> {
  const mediaDevices = navigator.mediaDevices;
  try {
    return await mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    });
  } catch (error) {
    // Algunos navegadores no soportan las restricciones de resolución/cámara:
    // reintentamos con la configuración más permisiva.
    if (errorName(error) === "OverconstrainedError") {
      return mediaDevices.getUserMedia({ video: true });
    }
    throw error;
  }
}

export function CameraCapture({ open, onOpenChange, onCapture, isLoading }: Props) {
  const [status, setStatus] = useState<CameraStatus>("starting");
  const [errorKind, setErrorKind] = useState<CameraErrorKind>("unknown");
  const [facingMode, setFacingMode] = useState<FacingMode>("environment");
  const [canSwitchCamera, setCanSwitchCamera] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [captured, setCaptured] = useState<{ url: string; file: File } | null>(null);
  // El stream activo vive en estado (no solo en un ref) para que el efecto que
  // lo conecta al <video> dependa de su identidad: al cambiar de cámara React
  // puede agrupar setStatus("starting") y setStatus("live") en un único render y
  // el estado nunca cambiaría, dejando el srcObject anterior congelado.
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const capturedUrlRef = useRef<string | null>(null);
  const fallbackInputRef = useRef<HTMLInputElement>(null);
  const openRef = useRef(open);

  useEffect(() => {
    openRef.current = open;
  }, [open]);

  const releaseCapturedUrl = useCallback(() => {
    if (capturedUrlRef.current) {
      URL.revokeObjectURL(capturedUrlRef.current);
      capturedUrlRef.current = null;
    }
  }, []);

  // Abre la cámara al mostrar el diálogo (y al cambiar de cámara). El flag
  // `cancelled` cubre el caso de cerrar mientras aún se espera el permiso.
  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    let stream: MediaStream | null = null;

    const start = async () => {
      releaseCapturedUrl();
      setCaptured(null);
      setActiveStream(null);
      setStatus("starting");
      setErrorKind("unknown");
      setVideoReady(false);
      setCanSwitchCamera(false);

      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        setErrorKind("unsupported");
        setStatus("error");
        return;
      }
      if (!window.isSecureContext) {
        setErrorKind("insecure");
        setStatus("error");
        return;
      }

      try {
        stream = await requestCameraStream(facingMode);
      } catch (error) {
        if (cancelled) return;
        setErrorKind(classifyError(error));
        setStatus("error");
        return;
      }

      if (cancelled) {
        stopStream(stream);
        stream = null;
        return;
      }

      setActiveStream(stream);
      setStatus("live");

      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (!cancelled) {
          setCanSwitchCamera(devices.filter((device) => device.kind === "videoinput").length > 1);
        }
      } catch {
        // enumerateDevices no es imprescindible: sin él simplemente no se
        // muestra el botón de cambiar cámara.
      }
    };

    void start();

    return () => {
      cancelled = true;
      stopStream(stream);
      stream = null;
      releaseCapturedUrl();
    };
  }, [open, facingMode, releaseCapturedUrl]);

  // El <video> solo existe en el estado `live`, así que conectamos el stream
  // después del render. La dependencia en `activeStream` es la que garantiza
  // que el cambio de cámara reenganche el video aunque `status` no cambie.
  useEffect(() => {
    const video = videoRef.current;
    if (status !== "live" || !video || !activeStream) return;
    video.srcObject = activeStream;
    return () => {
      video.srcObject = null;
    };
  }, [status, activeStream]);

  const handleShutter = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;

    const scale = Math.min(1, MAX_CAPTURE_EDGE / Math.max(video.videoWidth, video.videoHeight));
    const width = Math.max(1, Math.round(video.videoWidth * scale));
    const height = Math.max(1, Math.round(video.videoHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      setErrorKind("unknown");
      setStatus("error");
      return;
    }

    // La foto debe verse igual que el preview espejado de la cámara frontal.
    if (facingMode === "user") {
      context.translate(width, 0);
      context.scale(-1, 1);
    }
    context.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          if (openRef.current) {
            setErrorKind("unknown");
            setStatus("error");
          }
          return;
        }
        const url = URL.createObjectURL(blob);
        if (!openRef.current) {
          URL.revokeObjectURL(url);
          return;
        }
        releaseCapturedUrl();
        capturedUrlRef.current = url;
        setCaptured({
          url,
          file: new File([blob], `foto-${Date.now()}.jpg`, { type: "image/jpeg" }),
        });
        setStatus("captured");
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }, [facingMode, releaseCapturedUrl]);

  const handleRetry = () => {
    releaseCapturedUrl();
    setCaptured(null);
    setVideoReady(false);
    setStatus("live");
  };

  const closeDialog = useCallback(() => {
    // Vuelve a la cámara trasera para la próxima apertura. Se hace junto al
    // cierre (mismo render) para no arrancar un stream nuevo al desmontar.
    setFacingMode("environment");
    onOpenChange(false);
  }, [onOpenChange]);

  const handleUsePhoto = () => {
    if (!captured) return;
    const file = captured.file;
    closeDialog();
    onCapture(file);
  };

  const handleFallbackFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    closeDialog();
    onCapture(file);
  };

  const errorMessage = ERROR_MESSAGES[errorKind];

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (next) onOpenChange(true);
        else closeDialog();
      }}
    >
      <DialogContent
        className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg"
        showCloseButton={false}
      >
        <DialogHeader className="flex-row items-start justify-between gap-4">
          <div className="space-y-1">
            <DialogTitle>Tomar foto</DialogTitle>
            <DialogDescription>{DESCRIPTIONS[status]}</DialogDescription>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Cerrar cámara"
            onClick={closeDialog}
          >
            <X className="size-4" aria-hidden />
          </Button>
        </DialogHeader>

        {status === "starting" && (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed bg-muted/40 px-6 py-12 text-center">
            <Loader2 className="size-6 animate-spin text-muted-foreground" aria-hidden />
            <p role="status" className="text-sm text-muted-foreground">
              Pidiendo permiso para usar la cámara…
            </p>
          </div>
        )}

        {status === "live" && (
          <div className="space-y-4">
            <div className="overflow-hidden rounded-xl border bg-black">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                aria-label="Vista previa de la cámara"
                onLoadedMetadata={() => setVideoReady(true)}
                className={cn(
                  "aspect-[4/3] w-full object-contain",
                  facingMode === "user" && "-scale-x-100"
                )}
              />
            </div>
            <div className="flex items-center justify-between gap-3">
              <div className="flex-1">
                {canSwitchCamera && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    aria-label="Cambiar cámara"
                    onClick={() =>
                      setFacingMode((current) => (current === "environment" ? "user" : "environment"))
                    }
                  >
                    <SwitchCamera className="size-4" aria-hidden />
                    Cambiar cámara
                  </Button>
                )}
              </div>
              <Button
                type="button"
                aria-label="Capturar foto"
                onClick={handleShutter}
                disabled={!videoReady}
                className="size-14 shrink-0 rounded-full border-4 border-background bg-primary p-0 shadow-lg ring-2 ring-primary/40 hover:bg-primary/90"
              >
                <span
                  aria-hidden
                  className="size-10 rounded-full border-2 border-primary-foreground/80"
                />
              </Button>
            </div>
          </div>
        )}

        {status === "captured" && captured && (
          <div className="space-y-4">
            <div className="overflow-hidden rounded-xl border bg-muted">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={captured.url}
                alt="Vista previa de la foto capturada"
                className="max-h-[50vh] w-full object-contain"
              />
            </div>
            <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
              <Button type="button" variant="outline" onClick={handleRetry} disabled={isLoading}>
                <RefreshCw className="size-4" aria-hidden />
                Repetir
              </Button>
              <Button type="button" onClick={handleUsePhoto} disabled={isLoading}>
                <Check className="size-4" aria-hidden />
                Usar foto
              </Button>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="space-y-4">
            <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
              <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
              <p role="alert" className="text-sm">
                {errorMessage}
              </p>
            </div>
            <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
              <Button type="button" variant="outline" onClick={closeDialog}>
                Cancelar
              </Button>
              <Button type="button" onClick={() => fallbackInputRef.current?.click()}>
                <Camera className="size-4" aria-hidden />
                Abrir cámara del dispositivo
              </Button>
            </div>
            <input
              ref={fallbackInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={handleFallbackFile}
              aria-hidden
              tabIndex={-1}
            />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
