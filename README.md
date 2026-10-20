# Combina2

Combina 2 o más colores para generar una paleta. Identífica paletas de colores tomando una foto.

## Documentación

- [Backend](./Backend/README.md)

## Ejecutar con Docker

Desde la raíz: `docker compose up -d --build` y abre <http://localhost:3000>.
Consulta registros con `docker compose logs -f`; detén el stack con
`docker compose down`. El backend publica <http://localhost:8000>. La base
existente `backend/training.db` se monta directamente y conserva los datos;
el archivo debe existir antes de arrancar.
