from fastapi import status
from fastapi.testclient import TestClient

from src import app
from src.config import APICOnfig
from src.schemas.palette import ColorRole

client = TestClient(app)


def get_color_roles():
    """Listar colores"""
    response = client.get(f"{APICOnfig.prefix}/colors/roles")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()  # pyright: ignore[reportAny]

    assert isinstance(data, list)

    assert len(data) > 0

    # Validar si estan todos los enums posibles
    assert all(item in ColorRole.__members__.values() for item in data)
