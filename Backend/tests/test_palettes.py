# Backend/tests/test_palettes.py
from fastapi.testclient import TestClient
from src import app
from src.schemas.palette import HarmonyType, ColorRole

# Instanciamos el cliente de pruebas conectado a tu aplicación
client = TestClient(app)

def test_generate_harmonic_palette_complementary_success():
    """
    Caso de Uso 1: Éxito (200 OK).
    Generación de paleta complementaria rotando 180 grados el matiz.
    """
    payload = {
        "colors": ["#FF0000"], # Rojo puro
        "harmony": HarmonyType.COMPLEMENTARY.value
    }
    
    response = client.post("/api/v1/palettes/generate-harmonic", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == HarmonyType.COMPLEMENTARY.value
    assert len(data["colors"]) == 2
    
    # Validar Color Base
    assert data["colors"][0]["value"] == "#FF0000"
    assert data["colors"][0]["role"] == ColorRole.PRIMARY.value
    
    # Validar Complementario (Cyan)
    assert data["colors"][1]["value"] == "#00FFFF"
    assert data["colors"][1]["role"] == ColorRole.COMPLEMENTARY.value

def test_generate_harmonic_palette_triadic_success():
    """
    Caso de Uso 2: Éxito (200 OK).
    Generación de paleta triádica rotando +120 y +240 grados el matiz.
    """
    payload = {
        "colors": ["#00FF00"], # Verde puro
        "harmony": HarmonyType.TRIADIC.value
    }
    
    response = client.post("/api/v1/palettes/generate-harmonic", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["type"] == HarmonyType.TRIADIC.value
    assert len(data["colors"]) == 3
    
    # Validamos que los roles asignados sean correctos
    roles = [color["role"] for color in data["colors"]]
    assert ColorRole.PRIMARY.value in roles
    assert ColorRole.TRIADIC_1.value in roles
    assert ColorRole.TRIADIC_2.value in roles

def test_generate_harmonic_invalid_hex_format():
    """
    Caso de Uso 3: Error de lógica de negocio (400 Bad Request).
    El usuario envía un código hexadecimal inválido.
    """
    payload = {
        "colors": ["#FF"], # Incompleto
        "harmony": HarmonyType.COMPLEMENTARY.value
    }
    
    response = client.post("/api/v1/palettes/generate-harmonic", json=payload)
    
    assert response.status_code == 400
    assert "Código HEX inválido" in response.json()["detail"]

def test_generate_harmonic_missing_required_fields():
    """
    Caso de Uso 4: Error de validación de esquema (422 Unprocessable Entity).
    El usuario no envía la lista de colores obligatoria.
    """
    payload = {
        "harmony": HarmonyType.COMPLEMENTARY.value
    } # Falta el campo "colors"
    
    response = client.post("/api/v1/palettes/generate-harmonic", json=payload)
    
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "colors"]

def test_get_palette_options_success():
    """
    Criterio de Aceptación: Respuesta 200 OK y lista de opciones devuelta.
    Verifica que el endpoint devuelva las armonías soportadas basándose en el Enum.
    """
    response = client.get("/api/v1/palettes/options")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificamos que sea una lista
    assert isinstance(data, list)
    
    # Verificamos que contenga las opciones esperadas
    assert HarmonyType.COMPLEMENTARY.value in data
    assert HarmonyType.TRIADIC.value in data
    assert HarmonyType.ANALOGOUS.value in data
    assert HarmonyType.MONOCHROMATIC.value in data


def test_get_color_roles_success():
    """
    Criterio de Aceptación: Respuesta 200 OK y lista de roles devuelta.
    Verifica que el endpoint devuelva los roles de color soportados.
    """
    response = client.get("/api/v1/palettes/roles")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificamos que sea una lista
    assert isinstance(data, list)
    
    # Validamos que al menos contenga algunos de los roles clave
    assert ColorRole.PRIMARY.value in data
    assert ColorRole.COMPLEMENTARY.value in data
    assert ColorRole.ACCENT.value in data