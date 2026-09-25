# Backend/tests/test_identify.py
import io

from fastapi.testclient import TestClient
from PIL import Image

from src import app
from src.config import APICOnfig
from src.schemas.palette import ColorRole

client = TestClient(app)

IDENTIFY_URL = f"{APICOnfig.prefix}/pallete/identify"


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    # El formato JPEG no soporta transparencia: convertimos a RGB.
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def _post_image(content: bytes, filename: str = "image.png", content_type: str = "image/png"):
    return client.post(IDENTIFY_URL, files={"file": (filename, content, content_type)})


def test_identify_multiple_colors_success():
    """
    Caso de Uso 1: Éxito (200 OK).
    Imagen con varios colores distintos devuelve los predominantes ordenados.
    """
    image = Image.new("RGB", (200, 100), (255, 0, 0))  # Rojo
    image.paste((0, 0, 255), (100, 0, 200, 100))  # Azul en la mitad derecha

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    data = response.json()

    assert data["type"] == "predominant"
    assert isinstance(data["colors"], list)
    assert len(data["colors"]) == 2

    values = [color["value"] for color in data["colors"]]
    assert "#FF0000" in values
    assert "#0000FF" in values

    # Todos los roles deben ser "predominant"
    assert all(color["role"] == ColorRole.PREDOMINANT.value for color in data["colors"])


def test_identify_single_color():
    """
    Caso de Uso 2: Imagen de un solo color (rojo puro).
    No debe intentar crear 5 clusters: devuelve exactamente #FF0000.
    """
    image = Image.new("RGB", (100, 100), (255, 0, 0))

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    data = response.json()

    assert len(data["colors"]) == 1
    assert data["colors"][0]["value"] == "#FF0000"
    assert data["colors"][0]["role"] == ColorRole.PREDOMINANT.value


def test_identify_few_colors_adapts_clusters():
    """
    Caso de Uso 3: Imagen con pocos colores (3) adapta n_clusters automáticamente.
    """
    image = Image.new("RGB", (300, 100), (255, 0, 0))  # Rojo
    image.paste((0, 255, 0), (100, 0, 200, 100))  # Verde
    image.paste((0, 0, 255), (200, 0, 300, 100))  # Azul

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    data = response.json()

    assert len(data["colors"]) == 3
    values = [color["value"] for color in data["colors"]]
    assert set(values) == {"#FF0000", "#00FF00", "#0000FF"}


def test_identify_png_format():
    """
    Caso de Uso 4: PNG.
    """
    image = Image.new("RGB", (50, 50), (10, 20, 30))

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    assert response.json()["colors"][0]["value"] == "#0A141E"


def test_identify_rgba_with_transparency():
    """
    Caso de Uso 5: RGBA con transparencia parcial se compone sobre fondo blanco.
    Rojo al 50% de opacidad sobre blanco -> #FF7F7F.
    """
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    data = response.json()
    assert len(data["colors"]) == 1
    assert data["colors"][0]["value"] == "#FF7F7F"


def test_identify_rgba_fully_transparent_pixels_ignored():
    """
    Caso de Uso 5b: Los píxeles completamente transparentes no influyen en KMeans.
    La mitad derecha es transparente con un RGB "invisible" verde: el verde
    NO debe aparecer como predominante.
    """
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))  # Rojo visible
    transparent = Image.new("RGBA", (50, 100), (0, 255, 0, 0))  # Verde invisible
    image.paste(transparent, (50, 0))

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    values = [color["value"] for color in response.json()["colors"]]

    # El verde "invisible" no debe considerarse predominante.
    assert "#00FF00" not in values
    assert "#FF0000" in values
    assert "#FFFFFF" in values


def test_identify_grayscale():
    """
    Caso de Uso 6: Escala de grises se convierte a RGB correctamente.
    """
    image = Image.new("L", (100, 100), 128)

    response = _post_image(_png_bytes(image))

    assert response.status_code == 200
    data = response.json()
    assert len(data["colors"]) == 1
    assert data["colors"][0]["value"] == "#808080"


def test_identify_jpeg_format():
    """
    Caso de Uso 6b: JPEG. La imagen en formato JPEG se procesa correctamente.
    Nota: el codec JPEG es con pérdidas, por lo que se valida con tolerancia.
    """
    image = Image.new("RGB", (200, 100), (255, 0, 0))  # Rojo
    image.paste((0, 0, 255), (100, 0, 200, 100))  # Azul

    response = _post_image(
        _jpeg_bytes(image), filename="image.jpg", content_type="image/jpeg"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "predominant"

    # El codec JPEG introduce artefactos de color en el borde entre regiones,
    # por lo que no se exige un número exacto de colores, sino que el rojo y el
    # azul originales sean los dos colores más predominantes.
    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))

    def _close(actual: tuple[int, int, int], target: tuple[int, int, int], tol: int = 6):
        return all(abs(a - b) <= tol for a, b in zip(actual, target))

    top_two = [_hex_to_rgb(color["value"]) for color in data["colors"][:2]]

    assert any(_close(v, (255, 0, 0)) for v in top_two)
    assert any(_close(v, (0, 0, 255)) for v in top_two)


def test_identify_invalid_file():
    """
    Caso de Uso 7: Contenido de texto enviado como archivo -> 400.
    """
    response = _post_image(b"esto no es una imagen")

    assert response.status_code == 400


def test_identify_empty_file():
    """
    Caso de Uso 8: Archivo vacío -> 400.
    """
    response = _post_image(b"")

    assert response.status_code == 400
