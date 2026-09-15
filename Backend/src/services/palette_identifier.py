import io

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans

from src.schemas.palette import Color, ColorRole, PaletteResponse
from src.utils.color_converter import ColorConverter

# SISTEMA EXPERTO 2: Identificador de Paletas por Predominancia (K-Means)
MAX_CLUSTERS = 5
MAX_SIZE = (300, 300)
RANDOM_STATE = 42
N_INIT = 10


class ExpertSystemPaletteIdentifier:
    """
    Sistema Experto basado en Aprendizaje No Supervisado (K-Means).
    Extrae los colores predominantes de una imagen:
    Pillow (RGB) -> Resize -> NumPy -> KMeans -> centroides -> conteo por cluster
    -> ordenamiento por predominancia -> RGB -> HEX -> PaletteResponse.
    """

    def identify_palette(self, image_bytes: bytes) -> PaletteResponse:
        if not image_bytes:
            raise ValueError("El archivo está vacío")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                # Imágenes con transparencia: se componen sobre fondo blanco para
                # que los píxeles completamente transparentes no influyan en KMeans.
                if image.mode in ("RGBA", "LA", "PA"):
                    image = image.convert("RGBA")
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                else:
                    # Convertimos a RGB para tener exactamente 3 canales.
                    image = image.convert("RGB")
                # Reducimos imágenes grandes manteniendo la relación de aspecto.
                image.thumbnail(MAX_SIZE)
                # matriz [alto, ancho, 3] -> [cantidad_pixeles, 3]
                pixels = np.asarray(image).reshape(-1, 3)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("El archivo no es una imagen válida") from exc

        if pixels.shape[0] == 0:
            raise ValueError("La imagen no contiene píxeles")

        # Cantidad de colores RGB únicos tras el preprocesamiento.
        unique_colors = int(np.unique(pixels, axis=0).shape[0])

        # Nunca intentamos crear más clusters que colores únicos disponibles.
        n_clusters = min(MAX_CLUSTERS, unique_colors)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )
        kmeans.fit(pixels)

        # Contamos cuántos píxeles fueron asignados a cada cluster.
        counts = np.bincount(kmeans.labels_)

        # Ordenamos de mayor a menor predominancia (mayor cantidad de píxeles).
        order = np.argsort(counts)[::-1]

        colors: list[Color] = []
        for cluster_idx in order:
            centroid = kmeans.cluster_centers_[cluster_idx]
            # Los centroides pueden ser decimales: los convertimos a enteros RGB.
            rgb = tuple(
                int(round(max(0.0, min(255.0, channel)))) for channel in centroid
            )
            hex_value = ColorConverter.rgb_to_hex(rgb).upper()
            colors.append(Color(value=hex_value, role=ColorRole.PREDOMINANT.value))

        return PaletteResponse(type="predominant", colors=colors)
