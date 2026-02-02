# Healthy Record

Landing + formulario conectado a PostgreSQL, backend en FastAPI y panel de administración.

## 1) Estructura
- `index.html` sitio público
- `admin.html` panel de registros
- `main.py` API FastAPI
- `db/init/*.sql` scripts de base de datos
- `railway.json` despliegue en Railway
- `scripts/migrate.py` migraciones automáticas

## 2) Local (Docker + FastAPI)
```bash
docker compose up -d

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Abrir:
- Web: `http://127.0.0.1:8000`
- Admin: `http://127.0.0.1:8000/admin`

Credenciales admin en `.env`:
- `ADMIN_USER=admin`
- `ADMIN_PASSWORD=admin123`

## 3) Railway (deploy)
1. Subir repo a GitHub.
2. En Railway crear proyecto desde GitHub.
3. Agregar PostgreSQL en Railway.
4. Copiar `DATABASE_URL` al servicio Web.
5. Añadir variables en Railway: `ADMIN_USER`, `ADMIN_PASSWORD`.
6. Railway usa `railway.json` para ejecutar migraciones y levantar el servidor.

Comando de inicio:
```
python scripts/migrate.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```

## 4) Dominio propio
- Railway → Settings → Domains → Add Domain
- Crear CNAME/ALIAS en DNS apuntando al target de Railway.

## 5) Migraciones
Los SQL en `db/init` se ejecutan automáticamente en cada deploy.

## 6) CSV y filtros
- Panel admin permite filtrar por campos y exportar CSV.
- Ruta exportación: `/api/registrations.csv`.
