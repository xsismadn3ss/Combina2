from fastapi import Body


create_pallete_examples = Body(
    ...,
    openapi_examples={
        "Prueba Complementaria": {
            "summary": "Rojo -> Complementario",
            "description": "Se envía un rojo puro (#FF0000) esperando un Cyan (#00FFFF) como complementario.",
            "value": {"colors": ["#FF0000"], "harmony": "complementario"},
        },
        "Prueba Triádica": {
            "summary": "Verde -> Triada",
            "description": "Se envía un verde puro (#00FF00) para obtener Azul y Rojo.",
            "value": {"colors": ["#00FF00"], "harmony": "triada"},
        },
    },
)
