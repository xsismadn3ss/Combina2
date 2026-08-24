# Preparar entorno de desarrollo

## 1. Instalar UV

MacOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (powershell)

```powershell
winget install --id=astral-sh.uv  -e
```

> Más información en [Installation | uv](https://docs.astral.sh/uv/getting-started/installation/)

### ¿Qué es uv?

uv es un administrador de paquetes para python, desarrollado en Rust. Brina más control al trabajar en equipo porque crea un archivo `pyproject.toml` y `.python-version` que especifican las dependencias y la versión de python para el proyecto.

## 2. Instalar dependencias

Al usar uv no es necesario crear un entorno virtual manualmente, solo ejecuta `uv sync` y las dependencias del proyecto se instalar y automáticacmente creara el entorno virtual para el proyecto.

> **Nota:** Si tu entorno no cuenta con python 3.14, uv automáticamente lo instalara para que puede ser usado en el proyecto

## 3. Iniciar API

En el archivo `main.py` se encuentra el script para iniciar la API. Ejecuta `uv run main.py`.

Veras estos logs en consola:

```plain
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [33072] using StatReload
INFO:     Started server process [25196]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

```
