import os
import pathlib
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL no está configurada")

root = pathlib.Path(__file__).resolve().parents[1]
init_dir = root / "db" / "init"

if not init_dir.exists():
    raise SystemExit("No se encontró db/init")

sql_files = sorted(init_dir.glob("*.sql"))
if not sql_files:
    raise SystemExit("No hay archivos .sql en db/init")

with psycopg2.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        for sql_file in sql_files:
            sql = sql_file.read_text()
            if sql.strip():
                cur.execute(sql)

print("Migraciones aplicadas correctamente.")
