"""Motores clásicos (fallback).

Porteados de la rama `main` (sistemas expertos HSL + KMeans). Se usan cuando no hay
red neuronal activa (ver `inference.py` y `models/manifest.json`) y como maestros de
destilación en el entrenamiento (Fase 3+).
"""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans

from app.ml.color_spaces import ColorConverter
from app.schemas.palette import Color, ColorRole, HarmonyType, PaletteResponse

MAX_CLUSTERS = 10  # contrato: top-10 (en main era 5)
MAX_SIZE = (300, 300)
RANDOM_STATE = 42
N_INIT = 10


class ClassicPaletteGenerator:
    """Reglas geométricas fijas sobre el círculo cromático HSL."""

    def generate_palette(
        self, base_hex_colors: list[str], harmony: HarmonyType
    ) -> PaletteResponse:
        primary_hex = base_hex_colors[0]
        h, s, l = ColorConverter.hex_to_hsl(primary_hex)

        result_colors: list[Color] = [
            Color(value=primary_hex.upper(), role=ColorRole.PRIMARY.value)
        ]

        if harmony == HarmonyType.COMPLEMENTARY:
            comp_h = (h + 180.0) % 360.0
            comp_hex = ColorConverter.hsl_to_hex(comp_h, s, l)
            result_colors.append(
                Color(value=comp_hex.upper(), role=ColorRole.COMPLEMENTARY.value)
            )
        elif harmony == HarmonyType.TRIADIC:
            result_colors.append(
                Color(
                    value=ColorConverter.hsl_to_hex((h + 120.0) % 360.0, s, l).upper(),
                    role=ColorRole.TRIADIC_1.value,
                )
            )
            result_colors.append(
                Color(
                    value=ColorConverter.hsl_to_hex((h + 240.0) % 360.0, s, l).upper(),
                    role=ColorRole.TRIADIC_2.value,
                )
            )
        elif harmony == HarmonyType.ANALOGOUS:
            result_colors.append(
                Color(
                    value=ColorConverter.hsl_to_hex((h - 30.0) % 360.0, s, l).upper(),
                    role=ColorRole.ANALOGOUS_1.value,
                )
            )
            result_colors.append(
                Color(
                    value=ColorConverter.hsl_to_hex((h + 30.0) % 360.0, s, l).upper(),
                    role=ColorRole.ANALOGOUS_2.value,
                )
            )
        elif harmony == HarmonyType.MONOCHROMATIC:
            l_light = min(100.0, l + 25.0)
            l_dark = max(0.0, l - 25.0)
            result_colors.append(
                Color(
                    value=ColorConverter.hsl_to_hex(h, s, l_light).upper(),
                    role="light_tone",
                )
            )
            result_colors.append(
                Color(
                    value=ColorConverter.hsl_to_hex(h, s, l_dark).upper(),
                    role="dark_tone",
                )
            )

        return PaletteResponse(type=harmony.value, colors=result_colors, fallback=True)


class ClassicPaletteIdentifier:
    """KMeans sobre píxeles (Pillow -> RGB -> thumbnail -> KMeans -> HEX)."""

    def identify_palette(self, image_bytes: bytes) -> list[Color]:
        if not image_bytes:
            raise ValueError("El archivo está vacío")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                if image.mode in ("RGBA", "LA", "PA"):
                    image = image.convert("RGBA")
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                else:
                    image = image.convert("RGB")
                image.thumbnail(MAX_SIZE)
                pixels = np.asarray(image).reshape(-1, 3)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("El archivo no es una imagen válida") from exc

        if pixels.shape[0] == 0:
            raise ValueError("La imagen no contiene píxeles")

        unique_pixels, pixel_counts = np.unique(pixels, axis=0, return_counts=True)
        unique_colors = int(unique_pixels.shape[0])

        # KMeans is unnecessary for images with at most one distinct color per
        # requested cluster. Besides avoiding needless work, this makes 1x1,
        # tiny and low-color images independent of sklearn's degenerate
        # initialization/convergence behavior (notably duplicate samples).
        if unique_colors <= MAX_CLUSTERS:
            order = np.argsort(-pixel_counts, kind="stable")
            return [
                Color(
                    value=ColorConverter.rgb_to_hex(
                        tuple(int(channel) for channel in unique_pixels[idx])
                    ).upper(),
                    role=ColorRole.PREDOMINANT.value,
                    weight=float(pixel_counts[idx]) / float(pixel_counts.sum()),
                )
                for idx in order
            ]

        n_clusters = MAX_CLUSTERS

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )
        kmeans.fit(pixels)

        labels = kmeans.labels_
        assert labels is not None
        counts = np.bincount(labels)
        total = float(counts.sum())
        order = np.argsort(counts)[::-1]

        colors: list[Color] = []
        for cluster_idx in order:
            centroid = kmeans.cluster_centers_[cluster_idx]
            rgb: tuple[int, int, int] = (
                round(max(0.0, min(255.0, float(centroid[0])))),
                round(max(0.0, min(255.0, float(centroid[1])))),
                round(max(0.0, min(255.0, float(centroid[2])))),
            )
            colors.append(
                Color(
                    value=ColorConverter.rgb_to_hex(rgb).upper(),
                    role=ColorRole.PREDOMINANT.value,
                    weight=float(counts[cluster_idx]) / total,
                )
            )
        return colors
