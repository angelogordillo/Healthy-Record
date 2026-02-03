CREATE TABLE IF NOT EXISTS company_registrations (
  id BIGSERIAL PRIMARY KEY,
  empresa_nombre TEXT NOT NULL,
  empresa_web TEXT,
  colaboradores_rango TEXT NOT NULL,
  contacto_nombre TEXT NOT NULL,
  telefono_movil TEXT NOT NULL,
  email_corporativo TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_company_registrations_email
  ON company_registrations (email_corporativo);
