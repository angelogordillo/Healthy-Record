CREATE TABLE IF NOT EXISTS habit_registrations (
  id BIGSERIAL PRIMARY KEY,
  nombre TEXT NOT NULL,
  email TEXT NOT NULL,
  whatsapp TEXT NOT NULL,
  alimentacion TEXT NOT NULL,
  sueno_horas TEXT NOT NULL,
  sueno_calidad TEXT NOT NULL,
  hidratacion TEXT NOT NULL,
  ejercicio TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_habit_registrations_email
  ON habit_registrations (email);
