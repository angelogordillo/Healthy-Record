CREATE TABLE IF NOT EXISTS user_daily_entries (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL,
  entry_date DATE NOT NULL,
  ejercicio INT NOT NULL,
  agua INT NOT NULL,
  stress INT NOT NULL,
  calidad_sueno INT NOT NULL,
  horas_sueno NUMERIC(4,1) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (email, entry_date)
);

CREATE INDEX IF NOT EXISTS idx_user_daily_entries_email_date
  ON user_daily_entries (email, entry_date);
