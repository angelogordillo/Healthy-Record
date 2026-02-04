CREATE TABLE IF NOT EXISTS inbody_monthly_entries (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL,
  entry_month DATE NOT NULL,
  weight NUMERIC(6,2),
  bmi NUMERIC(5,2),
  body_fat_rate NUMERIC(5,2),
  muscle_total_pct NUMERIC(5,2),
  fat_brazo_izq NUMERIC(6,2),
  fat_brazo_der NUMERIC(6,2),
  fat_pierna_izq NUMERIC(6,2),
  fat_pierna_der NUMERIC(6,2),
  fat_tronco NUMERIC(6,2),
  muscle_brazo_izq NUMERIC(6,2),
  muscle_brazo_der NUMERIC(6,2),
  muscle_pierna_izq NUMERIC(6,2),
  muscle_pierna_der NUMERIC(6,2),
  muscle_tronco NUMERIC(6,2),
  file_path TEXT,
  extracted_json TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (email, entry_month)
);

CREATE INDEX IF NOT EXISTS idx_inbody_monthly_entries_email_month
  ON inbody_monthly_entries (email, entry_month);
