-- Postgres initialisation — run once on first container start.
-- Creates the two logical databases inside the single Postgres instance.
--
-- fall_detection : our tables (inference_log, feature_snapshot, fall_history,
--                  participant_session) — managed by Alembic migrations
-- mlflow         : MLflow internal tracking tables — managed by mlflow server

-- The 'fall_detection' database is already created by the POSTGRES_DB env var.
-- We only need to create 'mlflow' and grant permissions.

CREATE DATABASE mlflow
    WITH OWNER fall_user
    ENCODING 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE   = 'en_US.utf8';

GRANT ALL PRIVILEGES ON DATABASE mlflow TO fall_user;
GRANT ALL PRIVILEGES ON DATABASE fall_detection TO fall_user;
