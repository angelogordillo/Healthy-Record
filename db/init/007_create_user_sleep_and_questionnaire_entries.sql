CREATE TABLE IF NOT EXISTS user_sleep_habit_entries (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL,
  entry_date DATE NOT NULL,
  horas_cama NUMERIC(4,1) NOT NULL,
  despertares INT NOT NULL,
  energia_despertar INT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (email, entry_date)
);

CREATE INDEX IF NOT EXISTS idx_user_sleep_habit_entries_email_date
  ON user_sleep_habit_entries (email, entry_date);

CREATE TABLE IF NOT EXISTS user_questionnaire_entries (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL,
  entry_date DATE NOT NULL,
  alimentacion TEXT NOT NULL,
  sueno_horas TEXT NOT NULL,
  sueno_calidad TEXT NOT NULL,
  hidratacion TEXT NOT NULL,
  ejercicio TEXT NOT NULL,
  valoracion_general TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (email, entry_date)
);

CREATE INDEX IF NOT EXISTS idx_user_questionnaire_entries_email_date
  ON user_questionnaire_entries (email, entry_date);
