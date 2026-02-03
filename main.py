from pathlib import Path
from datetime import datetime, timedelta, date, timezone
import os
import base64
import csv
import io
import json
import secrets
import asyncio
import anyio
import psycopg2
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
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

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALIMENTACION_SCORES = {
    "Balanceada (3 comidas + snacks saludables)": 1.0,
    "Regular (2-3 comidas, pocos snacks)": 0.7,
    "Irregular (saltas comidas)": 0.4,
    "Alta en ultraprocesados": 0.2,
}
HIDRATACION_SCORES = {
    "2 litros o más": 1.0,
    "1.5 a 2 litros": 0.7,
    "1 a 1.5 litros": 0.5,
    "Menos de 1 litro": 0.2,
}
HIDRATACION_LITROS = {
    "2 litros o más": 2.0,
    "1.5 a 2 litros": 1.75,
    "1 a 1.5 litros": 1.25,
    "Menos de 1 litro": 0.75,
}
SUENO_HORAS_SCORES = {
    "7-8 horas": 1.0,
    "6-7 horas": 0.7,
    "Menos de 6 horas": 0.3,
    "Más de 8 horas": 0.8,
}
SUENO_HORAS_VAL = {
    "7-8 horas": 7.5,
    "6-7 horas": 6.5,
    "Menos de 6 horas": 5.5,
    "Más de 8 horas": 8.5,
}
SUENO_CALIDAD_SCORES = {
    "Buena (descanso continuo)": 1.0,
    "Regular (algunos despertares)": 0.7,
    "Baja (despertares frecuentes)": 0.3,
}
ACTIVIDAD_SCORES = {
    "3+ días (30-60 min)": 1.0,
    "1-2 días (30-60 min)": 0.7,
    "Solo caminatas ligeras": 0.4,
    "No realizo actividad física": 0.1,
}
PASOS_META = 7000
HIDRATACION_META = 2.0
SUENO_META = 7.0


def simulate_biometrics():
    return {
        "peso_kg": 87.5,
        "pasos_hoy": 6800,
        "pasos_meta": PASOS_META,
        "grasa": {
            "brazo_izq": 261.6,
            "brazo_der": 261.9,
            "pierna_izq": 227.6,
            "pierna_der": 226.8,
            "tronco": 315.6,
        },
        "musculo": {
            "brazo_izq": 93.9,
            "brazo_der": 94.2,
            "pierna_izq": 99.7,
            "pierna_der": 99.6,
            "tronco": 91.9,
        },
    }


def safe_avg(values: list[float]):
    return sum(values) / len(values) if values else None


def percent(value: float | None):
    if value is None:
        return None
    return round(value * 100)


def build_panel_data():
    now = datetime.now(timezone.utc)
    since_30 = now - timedelta(days=30)
    since_7 = now - timedelta(days=7)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM habit_registrations;")
            total_registrations = cur.fetchone()[0]
            cur.execute(
                """
                SELECT alimentacion, sueno_horas, sueno_calidad, hidratacion, ejercicio, created_at
                FROM habit_registrations
                WHERE created_at >= %s
                ORDER BY created_at DESC;
                """,
                (since_30,),
            )
            rows = cur.fetchall()

    alimentacion_scores = []
    hidratacion_scores = []
    sueno_scores = []
    actividad_scores = []
    hidratacion_litros = []
    sueno_horas_vals = []
    week_rows = []

    for row in rows:
        alimentacion, sueno_horas, sueno_calidad, hidratacion, ejercicio, created_at = row

        if created_at >= since_7:
            week_rows.append(row)

        if alimentacion in ALIMENTACION_SCORES:
            alimentacion_scores.append(ALIMENTACION_SCORES[alimentacion])
        if hidratacion in HIDRATACION_SCORES:
            hidratacion_scores.append(HIDRATACION_SCORES[hidratacion])
            hidratacion_litros.append(HIDRATACION_LITROS[hidratacion])
        if sueno_horas in SUENO_HORAS_SCORES or sueno_calidad in SUENO_CALIDAD_SCORES:
            horas_score = SUENO_HORAS_SCORES.get(sueno_horas)
            calidad_score = SUENO_CALIDAD_SCORES.get(sueno_calidad)
            scores = [val for val in [horas_score, calidad_score] if val is not None]
            if scores:
                sueno_scores.append(sum(scores) / len(scores))
        if sueno_horas in SUENO_HORAS_VAL:
            sueno_horas_vals.append(SUENO_HORAS_VAL[sueno_horas])
        if ejercicio in ACTIVIDAD_SCORES:
            actividad_scores.append(ACTIVIDAD_SCORES[ejercicio])

    alimentacion_pct = percent(safe_avg(alimentacion_scores))
    hidratacion_pct = percent(safe_avg(hidratacion_scores))
    sueno_pct = percent(safe_avg(sueno_scores))
    actividad_pct = percent(safe_avg(actividad_scores))
    hidratacion_l_prom = safe_avg(hidratacion_litros)
    sueno_h_prom = safe_avg(sueno_horas_vals)
    biometrics = simulate_biometrics()

    week_scores = {}
    hydration_by_day = {}
    sleep_by_day = {}
    activity_by_day = {}

    for row in week_rows:
        alimentacion, sueno_horas, sueno_calidad, hidratacion, ejercicio, created_at = row
        day_key = created_at.date()
        week_scores.setdefault(day_key, [])
        hydration_by_day.setdefault(day_key, [])
        sleep_by_day.setdefault(day_key, [])
        activity_by_day.setdefault(day_key, [])

        scores = []
        if alimentacion in ALIMENTACION_SCORES:
            scores.append(ALIMENTACION_SCORES[alimentacion])
        if hidratacion in HIDRATACION_SCORES:
            scores.append(HIDRATACION_SCORES[hidratacion])
            hydration_by_day[day_key].append(HIDRATACION_SCORES[hidratacion])
        if ejercicio in ACTIVIDAD_SCORES:
            scores.append(ACTIVIDAD_SCORES[ejercicio])
            activity_by_day[day_key].append(ACTIVIDAD_SCORES[ejercicio])

        horas_score = SUENO_HORAS_SCORES.get(sueno_horas)
        calidad_score = SUENO_CALIDAD_SCORES.get(sueno_calidad)
        sleep_scores = [val for val in [horas_score, calidad_score] if val is not None]
        if sleep_scores:
            sleep_value = sum(sleep_scores) / len(sleep_scores)
            scores.append(sleep_value)
            sleep_by_day[day_key].append(sleep_value)

        if scores:
            week_scores[day_key].append(sum(scores) / len(scores))

    week_series = []
    for offset in range(6, -1, -1):
        day = (now.date() - timedelta(days=offset))
        day_scores = week_scores.get(day, [])
        day_avg = safe_avg(day_scores)
        week_series.append(
            {
                "day": day.isoformat(),
                "score": percent(day_avg) if day_avg is not None else 0,
            }
        )

    low_hydration_days = len(
        [day for day, values in hydration_by_day.items() if safe_avg(values) is not None and safe_avg(values) < 0.7]
    )
    low_sleep_days = len(
        [day for day, values in sleep_by_day.items() if safe_avg(values) is not None and safe_avg(values) < 0.7]
    )
    low_activity_days = len(
        [day for day, values in activity_by_day.items() if safe_avg(values) is not None and safe_avg(values) < 0.7]
    )

    alerts = []
    if low_hydration_days >= 2:
        alerts.append(
            {
                "title": "Hidratacion baja",
                "detail": f"{low_hydration_days} dias por debajo de la meta",
                "tag": "Prioritario",
            }
        )
    if low_sleep_days >= 2:
        alerts.append(
            {
                "title": "Sueno irregular",
                "detail": f"{low_sleep_days} dias con sueno bajo",
                "tag": "Seguimiento",
            }
        )
    if low_activity_days >= 3:
        alerts.append(
            {
                "title": "Actividad limitada",
                "detail": f"{low_activity_days} dias con poca actividad",
                "tag": "Seguimiento",
            }
        )
    if not alerts:
        alerts.append(
            {
                "title": "Sin alertas criticas",
                "detail": "Los habitos se mantienen estables",
                "tag": "Ok",
            }
        )

    pasos_pct = (
        round(min(100, (biometrics["pasos_hoy"] / PASOS_META) * 100))
        if PASOS_META
        else None
    )

    return {
        "updated_at": now.isoformat() + "Z",
        "totals": {"registrations": total_registrations},
        "metrics": {
            "alimentacion_pct": alimentacion_pct,
            "hidratacion_pct": hidratacion_pct,
            "sueno_pct": sueno_pct,
            "actividad_pct": pasos_pct if pasos_pct is not None else actividad_pct,
        },
        "detail": {
            "hidratacion_litros": round(hidratacion_l_prom, 1) if hidratacion_l_prom is not None else None,
            "sueno_horas": round(sueno_h_prom, 1) if sueno_h_prom is not None else None,
            "pasos_hoy": biometrics["pasos_hoy"],
            "pasos_meta": PASOS_META,
            "peso_kg": biometrics["peso_kg"],
        },
        "biometrics": {
            "grasa": biometrics["grasa"],
            "musculo": biometrics["musculo"],
        },
        "week": week_series,
        "alerts": alerts,
    }

class HabitRegistration(BaseModel):
    nombre: str
    email: EmailStr
    whatsapp: str
    alimentacion: str
    sueno_horas: str
    sueno_calidad: str
    hidratacion: str
    ejercicio: str


class CompanyRegistration(BaseModel):
    empresa_nombre: str
    empresa_web: str | None = None
    colaboradores_rango: str
    contacto_nombre: str
    telefono_movil: str
    email_corporativo: EmailStr
    password: str


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

@app.get("/api/panel")
def panel_snapshot():
    try:
        return build_panel_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/panel/stream")
async def panel_stream():
    async def event_generator():
        while True:
            data = await anyio.to_thread.run_sync(build_panel_data)
            yield f"event: panel\ndata: {json.dumps(data)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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


@app.post("/api/company-register")
def company_register(payload: CompanyRegistration):
    try:
        password_hash = pwd_context.hash(payload.password)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO company_registrations
                      (empresa_nombre, empresa_web, colaboradores_rango, contacto_nombre, telefono_movil, email_corporativo, password_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        payload.empresa_nombre,
                        payload.empresa_web,
                        payload.colaboradores_rango,
                        payload.contacto_nombre,
                        payload.telefono_movil,
                        payload.email_corporativo,
                        password_hash,
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
            row = list(row)
            # Force Excel to treat certain fields as text
            row[2] = f"\t{row[2]}"  # colaboradores_rango
            row[4] = f"\t{row[4]}"  # telefono_movil
            writer.writerow(row)
        csv_data = output.getvalue()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=registros.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/companies")
def list_companies(_: bool = Depends(require_basic_auth)):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT empresa_nombre, empresa_web, colaboradores_rango, contacto_nombre, telefono_movil,
                           email_corporativo, created_at
                    FROM company_registrations
                    ORDER BY created_at DESC
                    LIMIT 200;
                    """
                )
                rows = cur.fetchall()
        return [
            {
                "empresa_nombre": row[0],
                "empresa_web": row[1],
                "colaboradores_rango": row[2],
                "contacto_nombre": row[3],
                "telefono_movil": row[4],
                "email_corporativo": row[5],
                "created_at": row[6].isoformat(),
            }
            for row in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/companies.csv")
def export_companies_csv(_: bool = Depends(require_basic_auth)):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT empresa_nombre, empresa_web, colaboradores_rango, contacto_nombre, telefono_movil,
                           email_corporativo, created_at
                    FROM company_registrations
                    ORDER BY created_at DESC
                    LIMIT 2000;
                    """
                )
                rows = cur.fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "empresa_nombre",
                "empresa_web",
                "colaboradores_rango",
                "contacto_nombre",
                "telefono_movil",
                "email_corporativo",
                "created_at",
            ]
        )
        for row in rows:
            writer.writerow(row)
        csv_data = output.getvalue()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=empresas.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
