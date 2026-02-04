ALTER TABLE habit_registrations
ADD COLUMN IF NOT EXISTS password_hash TEXT;
