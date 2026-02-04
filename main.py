from pathlib import Path
from datetime import datetime, timedelta, date, timezone
import os
import base64
import csv
import io
import json
import secrets
import hmac
import hashlib
import time
import re
import asyncio
import anyio
import psycopg2
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
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
APP_SECRET = os.getenv("APP_SECRET", "dev-secret")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", Path(__file__).parent / "uploads"))

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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
        "bmi": 27.0,
        "body_score": 64,
        "body_fat_rate": 30.7,
        "visceral_fat_grade": 11,
        "bmr_kcal": 1678,
        "fat_free_weight": 60.8,
        "subcutaneous_fat": 21.9,
        "smi": 8.3,
        "body_age": 51,
        "whr": 0.91,
        "target_weight": 71.4,
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


def extract_section(text: str, start_label: str, end_label: str | None = None):
    lower = text.lower()
    start = lower.find(start_label.lower())
    if start == -1:
        return ""
    end = lower.find(end_label.lower(), start) if end_label else -1
    if end == -1:
        return text[start:]
    return text[start:end]


def extract_number(patterns: list[str], text: str):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def extract_percent_by_label(text: str, label: str):
    for line in text.splitlines():
        if label.lower() in line.lower():
            numbers = re.findall(r"([0-9]+(?:\.[0-9]+)?)", line)
            if numbers:
                return float(numbers[-1])
    return None


def parse_inbody_text(text: str):
    body_fat_rate = extract_number(
        [
            r"Body fat rate\s*([0-9]+(?:\.[0-9]+)?)",
            r"Body fat rate\(\%\)\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        text,
    )
    bmi = extract_number([r"\bBMI\b\s*([0-9]+(?:\.[0-9]+)?)"], text)
    weight = extract_number([r"Weight\(kg\)\s*([0-9]+(?:\.[0-9]+)?)", r"Weight\s*([0-9]+(?:\.[0-9]+)?)"], text)
    muscle_total_pct = extract_number(
        [r"\bMuscle\b\s+[0-9]+(?:\.[0-9]+)?\s*\([^\)]+\)\s*([0-9]+(?:\.[0-9]+)?)"],
        text,
    )

    fat_section = extract_section(text, "Segmental fat analysis", "Muscle balance")
    muscle_section = extract_section(text, "Muscle balance", "Bioelectrical impedance")

    fat = {
        "brazo_izq": extract_percent_by_label(fat_section, "Left Arm"),
        "brazo_der": extract_percent_by_label(fat_section, "Right Arm"),
        "pierna_izq": extract_percent_by_label(fat_section, "Left Leg"),
        "pierna_der": extract_percent_by_label(fat_section, "Right Leg"),
        "tronco": extract_percent_by_label(fat_section, "Trunk"),
    }
    muscle = {
        "brazo_izq": extract_percent_by_label(muscle_section, "Left Arm"),
        "brazo_der": extract_percent_by_label(muscle_section, "Right Arm"),
        "pierna_izq": extract_percent_by_label(muscle_section, "Left Leg"),
        "pierna_der": extract_percent_by_label(muscle_section, "Right Leg"),
        "tronco": extract_percent_by_label(muscle_section, "Trunk"),
    }

    return {
        "weight": weight,
        "bmi": bmi,
        "body_fat_rate": body_fat_rate,
        "muscle_total_pct": muscle_total_pct,
        "fat": fat,
        "muscle": muscle,
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
            "bmi": biometrics["bmi"],
            "body_score": biometrics["body_score"],
            "body_fat_rate": biometrics["body_fat_rate"],
            "visceral_fat_grade": biometrics["visceral_fat_grade"],
            "bmr_kcal": biometrics["bmr_kcal"],
            "fat_free_weight": biometrics["fat_free_weight"],
            "subcutaneous_fat": biometrics["subcutaneous_fat"],
            "smi": biometrics["smi"],
            "body_age": biometrics["body_age"],
            "whr": biometrics["whr"],
            "target_weight": biometrics["target_weight"],
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
    password: str


class HabitLogin(BaseModel):
    email: EmailStr
    password: str


class DailyEntry(BaseModel):
    entry_date: date | None = None
    ejercicio: int
    agua: int
    stress: int
    calidad_sueno: int
    horas_sueno: float


class MonthlyInbodyEntry(BaseModel):
    entry_month: date


class ManualInbodyEntry(BaseModel):
    entry_month: date
    weight: float | None = None
    bmi: float | None = None
    body_fat_rate: float | None = None
    muscle_total_pct: float | None = None
    fat_brazo_izq: float | None = None
    fat_brazo_der: float | None = None
    fat_pierna_izq: float | None = None
    fat_pierna_der: float | None = None
    fat_tronco: float | None = None
    muscle_brazo_izq: float | None = None
    muscle_brazo_der: float | None = None
    muscle_pierna_izq: float | None = None
    muscle_pierna_der: float | None = None
    muscle_tronco: float | None = None


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


def create_token(email: str, expires_in: int = 60 * 60 * 2):
    payload = {"email": email, "exp": int(time.time()) + expires_in}
    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    signature = hmac.new(APP_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
    return f"{token}.{signature}"


def verify_token(token: str):
    try:
        encoded_payload, signature = token.split(".", 1)
        padding = "=" * (-len(encoded_payload) % 4)
        payload_bytes = base64.urlsafe_b64decode(encoded_payload + padding)
        expected = hmac.new(APP_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        if not secrets.compare_digest(signature, expected):
            return None
        payload = json.loads(payload_bytes.decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def require_panel_auth(request: Request):
    auth = request.headers.get("Authorization", "")
    token = None
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        token = request.query_params.get("token")
    payload = verify_token(token) if token else None
    if not payload:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return payload


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
def panel_snapshot(_: bool = Depends(require_panel_auth)):
    try:
        return build_panel_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/panel/stream")
async def panel_stream(_: dict = Depends(require_panel_auth)):
    async def event_generator():
        while True:
            data = await anyio.to_thread.run_sync(build_panel_data)
            yield f"event: panel\ndata: {json.dumps(data)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/panel/entries")
def create_daily_entry(payload: DailyEntry, auth: dict = Depends(require_panel_auth)):
    try:
        entry_date = payload.entry_date or datetime.now(timezone.utc).date()
        if entry_date > datetime.now(timezone.utc).date():
            raise HTTPException(status_code=400, detail="Invalid date")
        if not all(1 <= value <= 7 for value in [payload.ejercicio, payload.agua, payload.stress, payload.calidad_sueno]):
            raise HTTPException(status_code=400, detail="Values must be between 1 and 7")
        if payload.horas_sueno <= 0 or payload.horas_sueno > 24:
            raise HTTPException(status_code=400, detail="Invalid sleep hours")

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_daily_entries
                      (email, entry_date, ejercicio, agua, stress, calidad_sueno, horas_sueno)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email, entry_date) DO NOTHING
                    RETURNING id;
                    """,
                    (
                        auth["email"],
                        entry_date,
                        payload.ejercicio,
                        payload.agua,
                        payload.stress,
                        payload.calidad_sueno,
                        payload.horas_sueno,
                    ),
                )
                new_id = cur.fetchone()
        if not new_id:
            raise HTTPException(status_code=409, detail="Entry already exists")
        return {"ok": True, "date": entry_date.isoformat()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/panel/entries")
def list_daily_entries(days: int = 90, auth: dict = Depends(require_panel_auth)):
    try:
        safe_days = max(7, min(days, 365))
        since = datetime.now(timezone.utc).date() - timedelta(days=safe_days - 1)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT entry_date, ejercicio, agua, stress, calidad_sueno, horas_sueno
                    FROM user_daily_entries
                    WHERE email = %s AND entry_date >= %s
                    ORDER BY entry_date ASC;
                    """,
                    (auth["email"], since),
                )
                rows = cur.fetchall()
        entries = [
            {
                "date": row[0].isoformat(),
                "ejercicio": row[1],
                "agua": row[2],
                "stress": row[3],
                "calidad_sueno": row[4],
                "horas_sueno": float(row[5]),
            }
            for row in rows
        ]
        return {"entries": entries}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/panel/inbody")
def upload_inbody_report(
    month: str | None = None,
    file: UploadFile = File(...),
    auth: dict = Depends(require_panel_auth),
):
    try:
        if file.content_type not in ["application/pdf"] and not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type")
        if month:
            try:
                entry_month = datetime.strptime(month, "%Y-%m").date().replace(day=1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid month format")
        else:
            entry_month = datetime.now(timezone.utc).date().replace(day=1)

        contents = file.file.read()
        filename = f"inbody_{auth['email'].replace('@', '_')}_{entry_month.isoformat()}.pdf"
        save_path = UPLOAD_DIR / filename
        with open(save_path, "wb") as out_file:
            out_file.write(contents)

        try:
            from PyPDF2 import PdfReader
        except Exception:
            raise HTTPException(status_code=500, detail="PDF parser not available")

        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        parsed = parse_inbody_text(text)
        if parsed["weight"] is None or parsed["bmi"] is None or parsed["body_fat_rate"] is None:
            return JSONResponse(
                status_code=422,
                content={"detail": "Unable to extract required values", "extracted": parsed},
            )

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inbody_monthly_entries
                      (email, entry_month, weight, bmi, body_fat_rate, muscle_total_pct,
                       fat_brazo_izq, fat_brazo_der, fat_pierna_izq, fat_pierna_der, fat_tronco,
                       muscle_brazo_izq, muscle_brazo_der, muscle_pierna_izq, muscle_pierna_der, muscle_tronco,
                       file_path, extracted_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email, entry_month) DO NOTHING
                    RETURNING id;
                    """,
                    (
                        auth["email"],
                        entry_month,
                        parsed["weight"],
                        parsed["bmi"],
                        parsed["body_fat_rate"],
                        parsed["muscle_total_pct"],
                        parsed["fat"]["brazo_izq"],
                        parsed["fat"]["brazo_der"],
                        parsed["fat"]["pierna_izq"],
                        parsed["fat"]["pierna_der"],
                        parsed["fat"]["tronco"],
                        parsed["muscle"]["brazo_izq"],
                        parsed["muscle"]["brazo_der"],
                        parsed["muscle"]["pierna_izq"],
                        parsed["muscle"]["pierna_der"],
                        parsed["muscle"]["tronco"],
                        str(save_path),
                        json.dumps(parsed),
                    ),
                )
                new_id = cur.fetchone()
        if not new_id:
            raise HTTPException(status_code=409, detail="Entry already exists")
        return {"ok": True, "month": entry_month.isoformat()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/panel/inbody/manual")
def create_inbody_manual(payload: ManualInbodyEntry, auth: dict = Depends(require_panel_auth)):
    try:
        entry_month = payload.entry_month.replace(day=1)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inbody_monthly_entries
                      (email, entry_month, weight, bmi, body_fat_rate, muscle_total_pct,
                       fat_brazo_izq, fat_brazo_der, fat_pierna_izq, fat_pierna_der, fat_tronco,
                       muscle_brazo_izq, muscle_brazo_der, muscle_pierna_izq, muscle_pierna_der, muscle_tronco,
                       file_path, extracted_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email, entry_month) DO NOTHING
                    RETURNING id;
                    """,
                    (
                        auth["email"],
                        entry_month,
                        payload.weight,
                        payload.bmi,
                        payload.body_fat_rate,
                        payload.muscle_total_pct,
                        payload.fat_brazo_izq,
                        payload.fat_brazo_der,
                        payload.fat_pierna_izq,
                        payload.fat_pierna_der,
                        payload.fat_tronco,
                        payload.muscle_brazo_izq,
                        payload.muscle_brazo_der,
                        payload.muscle_pierna_izq,
                        payload.muscle_pierna_der,
                        payload.muscle_tronco,
                        None,
                        json.dumps(payload.dict()),
                    ),
                )
                new_id = cur.fetchone()
        if not new_id:
            raise HTTPException(status_code=409, detail="Entry already exists")
        return {"ok": True, "month": entry_month.isoformat()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/panel/inbody")
def list_inbody_reports(months: int = 12, auth: dict = Depends(require_panel_auth)):
    try:
        safe_months = max(1, min(months, 36))
        since = (datetime.now(timezone.utc).date().replace(day=1) - timedelta(days=31 * (safe_months - 1)))
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT entry_month, weight, bmi, body_fat_rate, muscle_total_pct,
                           fat_brazo_izq, fat_brazo_der, fat_pierna_izq, fat_pierna_der, fat_tronco,
                           muscle_brazo_izq, muscle_brazo_der, muscle_pierna_izq, muscle_pierna_der, muscle_tronco
                    FROM inbody_monthly_entries
                    WHERE email = %s AND entry_month >= %s
                    ORDER BY entry_month ASC;
                    """,
                    (auth["email"], since),
                )
                rows = cur.fetchall()
        entries = [
            {
                "month": row[0].isoformat(),
                "weight": row[1],
                "bmi": row[2],
                "body_fat_rate": row[3],
                "muscle_total_pct": row[4],
                "fat_brazo_izq": row[5],
                "fat_brazo_der": row[6],
                "fat_pierna_izq": row[7],
                "fat_pierna_der": row[8],
                "fat_tronco": row[9],
                "muscle_brazo_izq": row[10],
                "muscle_brazo_der": row[11],
                "muscle_pierna_izq": row[12],
                "muscle_pierna_der": row[13],
                "muscle_tronco": row[14],
            }
            for row in rows
        ]
        return {"entries": entries}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/register")
def register(payload: HabitRegistration):
    try:
        if not payload.password or len(payload.password) < 6:
            raise HTTPException(status_code=400, detail="Password too short")
        password_hash = pwd_context.hash(payload.password)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO habit_registrations
                      (nombre, email, whatsapp, alimentacion, sueno_horas, sueno_calidad, hidratacion, ejercicio, password_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        password_hash,
                    ),
                )
                new_id = cur.fetchone()[0]
        return {"ok": True, "id": new_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/panel-login")
def panel_login(payload: HabitLogin):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT password_hash
                    FROM habit_registrations
                    WHERE email = %s
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    (payload.email,),
                )
                row = cur.fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not pwd_context.verify(payload.password, row[0]):
            raise HTTPException(status_code=401, detail="Unauthorized")
        token = create_token(payload.email)
        return {"ok": True, "token": token}
    except HTTPException:
        raise
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
