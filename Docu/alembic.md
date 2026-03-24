## Alembic

Alembic is a **database migration tool** for Python. It manages changes to your PostgreSQL schema over time.

---

**The problem it solves:**

You have Python code that defines your tables (in `shared/db/models.py` using SQLAlchemy). But the actual PostgreSQL database doesn't know about those tables yet — someone has to go create them. And later, if you add a new column or table, someone has to update the database too.

Alembic does that automatically and safely.

---

**How it works in this project:**

```
shared/db/models.py          ← defines the tables in Python
       │
       │  alembic upgrade head
       ▼
PostgreSQL (localhost:5432)  ← creates the actual tables
   inference_log
   feature_snapshot
   participant_session
   api_request_log
```

The migration files live in `shared/db/migrations/versions/`. Each one is a numbered script that says "add this table" or "add this column". Alembic tracks which migrations have already been applied so it never runs the same one twice.

---

**The two commands you'll actually use:**

```
# Apply all pending migrations (create/update tables)
alembic upgrade head

# See what version the DB is currently at
alembic current
```

`upgrade head` means "run every migration up to the latest one." You run this once when setting up the project, and again any time new migrations are added.

---

**Analogy:** Think of it like `git` but for your database schema. Each migration file is a commit — it records exactly what changed and when. You can upgrade or roll back to any point in history.