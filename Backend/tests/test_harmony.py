from starlette.testclient import TestClient

from src import app
from src.config import APICOnfig
from src.schemas.palette import HarmonyType

client = TestClient(app)


def test_get_harmony():
    response = client.get(f"{APICOnfig.prefix}/harmony/")

    assert response.status_code == 200
    data = response.json()

    # Validar si es una lista
    assert isinstance(data, list)

    assert HarmonyType.COMPLEMENTARY in data
    assert HarmonyType.TRIADIC in data

    # Validar si estan todos los enum
    assert all(item in data for item in HarmonyType.__members__.values())
