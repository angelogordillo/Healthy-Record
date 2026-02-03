from pathlib import Path
import os
import base64
import csv
import io
import secrets
import psycopg2
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "crear_mi_registro")
DB_USER = os.getenv("DB_USER", "agordillo")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234567890")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class HabitRegistration(BaseModel):
    nombre: str
    email: EmailStr
    whatsapp: str
    alimentacion: str
    sueno_horas: str
    sueno_calidad: str
    hidratacion: str
    ejercicio: str


def get_conn():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def require_basic_auth(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Basic "):
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
            detail="Unauthorized",
        )
    try:
        decoded = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
    except Exception:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
            detail="Unauthorized",
        )
    username, _, password = decoded.partition(":")
    if not username or not secrets.compare_digest(username, ADMIN_USER):
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
            detail="Unauthorized",
        )
    if not secrets.compare_digest(password, ADMIN_PASSWORD):
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
            detail="Unauthorized",
        )
    return True


@app.get("/")
def index():
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path)

@app.get("/admin")
def admin(_: bool = Depends(require_basic_auth)):
    admin_path = Path(__file__).parent / "admin.html"
    return FileResponse(admin_path)

@app.get("/panel")
def panel():
    panel_path = Path(__file__).parent / "panel.html"
    return FileResponse(panel_path)


@app.post("/api/register")
def register(payload: HabitRegistration):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO habit_registrations
                      (nombre, email, whatsapp, alimentacion, sueno_horas, sueno_calidad, hidratacion, ejercicio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        payload.nombre,
                        payload.email,
                        payload.whatsapp,
                        payload.alimentacion,
                        payload.sueno_horas,
                        payload.sueno_calidad,
                        payload.hidratacion,
                        payload.ejercicio,
                    ),
                )
                new_id = cur.fetchone()[0]
        return {"ok": True, "id": new_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

def build_filters(
    nombre: str | None,
    email: str | None,
    whatsapp: str | None,
    alimentacion: str | None,
    sueno_horas: str | None,
    sueno_calidad: str | None,
    hidratacion: str | None,
    ejercicio: str | None,
    date_from: str | None,
    date_to: str | None,
):
    conditions = []
    values: list[str] = []

    if nombre:
        conditions.append("LOWER(nombre) LIKE LOWER(%s)")
        values.append(f"%{nombre}%")
    if email:
        conditions.append("LOWER(email) LIKE LOWER(%s)")
        values.append(f"%{email}%")
    if whatsapp:
        conditions.append("whatsapp LIKE %s")
        values.append(f"%{whatsapp}%")
    if alimentacion:
        conditions.append("alimentacion = %s")
        values.append(alimentacion)
    if sueno_horas:
        conditions.append("sueno_horas = %s")
        values.append(sueno_horas)
    if sueno_calidad:
        conditions.append("sueno_calidad = %s")
        values.append(sueno_calidad)
    if hidratacion:
        conditions.append("hidratacion = %s")
        values.append(hidratacion)
    if ejercicio:
        conditions.append("ejercicio = %s")
        values.append(ejercicio)
    if date_from:
        conditions.append("created_at >= %s")
        values.append(date_from)
    if date_to:
        conditions.append("created_at <= %s")
        values.append(date_to)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    return where_clause, values


def fetch_registrations(where_clause: str, values: list[str], limit: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT nombre, email, whatsapp, alimentacion, sueno_horas, sueno_calidad, hidratacion, ejercicio, created_at
                FROM habit_registrations
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (*values, limit),
            )
            return cur.fetchall()

@app.get("/api/registrations")
def list_registrations(
    nombre: str | None = None,
    email: str | None = None,
    whatsapp: str | None = None,
    alimentacion: str | None = None,
    sueno_horas: str | None = None,
    sueno_calidad: str | None = None,
    hidratacion: str | None = None,
    ejercicio: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 200,
    _: bool = Depends(require_basic_auth),
):
    try:
        safe_limit = max(1, min(limit, 1000))
        where_clause, values = build_filters(
            nombre,
            email,
            whatsapp,
            alimentacion,
            sueno_horas,
            sueno_calidad,
            hidratacion,
            ejercicio,
            date_from,
            date_to,
        )
        rows = fetch_registrations(where_clause, values, safe_limit)
        return [
            {
                "nombre": row[0],
                "email": row[1],
                "whatsapp": row[2],
                "alimentacion": row[3],
                "sueno_horas": row[4],
                "sueno_calidad": row[5],
                "hidratacion": row[6],
                "ejercicio": row[7],
                "created_at": row[8].isoformat(),
            }
            for row in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/registrations.csv")
def export_registrations_csv(
    nombre: str | None = None,
    email: str | None = None,
    whatsapp: str | None = None,
    alimentacion: str | None = None,
    sueno_horas: str | None = None,
    sueno_calidad: str | None = None,
    hidratacion: str | None = None,
    ejercicio: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 1000,
    _: bool = Depends(require_basic_auth),
):
    try:
        safe_limit = max(1, min(limit, 5000))
        where_clause, values = build_filters(
            nombre,
            email,
            whatsapp,
            alimentacion,
            sueno_horas,
            sueno_calidad,
            hidratacion,
            ejercicio,
            date_from,
            date_to,
        )
        rows = fetch_registrations(where_clause, values, safe_limit)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "nombre",
                "email",
                "whatsapp",
                "alimentacion",
                "sueno_horas",
                "sueno_calidad",
                "hidratacion",
                "ejercicio",
                "created_at",
            ]
        )
        for row in rows:
            writer.writerow(row)
        csv_data = output.getvalue()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=registros.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
