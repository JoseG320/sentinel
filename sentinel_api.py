"""
Sentinel – FastAPI Backend
===========================
Provides CRUD endpoints for cameras, zones, and events.
This is the only service that talks directly to PostgreSQL.

Requirements:
    pip install fastapi uvicorn psycopg2-binary python-dotenv

Run with:
    uvicorn sentinel_api:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs

The docs are also available in the app's settings page.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
import os
import random
import bcrypt
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Sentinel API", version="1.0.0")

# Allow Streamlit (any localhost origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DATABASE
def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "sentinel"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )

def dict_cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# SCHEMA INIT (runs on startup)
@app.on_event("startup")
def init_db():
    conn = get_db()
    with conn.cursor() as cur:
        # cameras first — zones references it
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id         SERIAL PRIMARY KEY,
                name       TEXT NOT NULL,
                stream_url TEXT NOT NULL,
                location   TEXT,
                active     BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL,
                alert_level TEXT NOT NULL CHECK (alert_level IN ('HIGH','MEDIUM','LOW')),
                x1          REAL NOT NULL,
                y1          REAL NOT NULL,
                x2          REAL NOT NULL,
                y2          REAL NOT NULL,
                camera_id   INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
                active      BOOLEAN DEFAULT TRUE,
                created_at  TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id             SERIAL PRIMARY KEY,
                zone_id        INTEGER REFERENCES zones(id) ON DELETE SET NULL,
                detected_at    TIMESTAMP NOT NULL DEFAULT NOW(),
                detection_type TEXT NOT NULL CHECK (detection_type IN ('Person Detected','Motion Only')),
                alert_level    TEXT NOT NULL CHECK (alert_level IN ('HIGH','MEDIUM','LOW')),
                snapshot_path  TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'admin'
                                  CHECK (role IN ('admin','viewer')),
                created_at    TIMESTAMP DEFAULT NOW()
            );
        """)
        # Migration safety net for existing databases
        cur.execute("""
            ALTER TABLE zones ADD COLUMN IF NOT EXISTS
            camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL;
        """)
        cur.execute("""
            ALTER TABLE zones ADD COLUMN IF NOT EXISTS
            active BOOLEAN DEFAULT TRUE;
        """)
        conn.commit()
    conn.close()
    print("[sentinel_api] Database initialised.")


# PYDANTIC MODELS
class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    role:     str = "admin"

class CameraCreate(BaseModel):
    name:       str
    stream_url: str
    location:   Optional[str] = None

class CameraUpdate(BaseModel):
    name:       Optional[str] = None
    stream_url: Optional[str] = None
    location:   Optional[str] = None
    active:     Optional[bool] = None

class ZoneCreate(BaseModel):
    name:        str
    alert_level: str
    x1:          float
    y1:          float
    x2:          float
    y2:          float
    camera_id:   Optional[int] = None

class ZoneUpdate(BaseModel):
    name:        Optional[str]   = None
    alert_level: Optional[str]   = None
    x1:          Optional[float] = None
    y1:          Optional[float] = None
    x2:          Optional[float] = None
    y2:          Optional[float] = None

class EventCreate(BaseModel):
    zone_id:        Optional[int] = None
    detection_type: str
    alert_level:    str
    snapshot_path:  Optional[str] = None

# HEALTH
@app.get("/health")
def health():
    try:
        conn = get_db()
        conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

# AUTH
@app.post("/auth/setup", status_code=201)
def setup_admin(body: UserCreate):
    """
    Creates the first admin user. Fails if any users already exist.
    Used for first-time setup only.
    """
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users;")
        if cur.fetchone()[0] > 0:
            conn.close()
            raise HTTPException(status_code=409, detail="Setup already complete. Users already exist.")
        password_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
        cur.execute("""
            INSERT INTO users (username, password_hash, role)
            VALUES (%s, %s, 'admin');
        """, (body.username.strip(), password_hash))
        conn.commit()
    conn.close()
    return {"message": f"Admin user '{body.username}' created successfully."}

@app.post("/auth/login")
def login(body: LoginRequest):
    """Validates credentials. Returns user info on success."""
    conn = get_db()
    with dict_cursor(conn) as cur:
        cur.execute("SELECT * FROM users WHERE username = %s;", (body.username.strip(),))
        user = cur.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    if not bcrypt.checkpw(body.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {
        "success":  True,
        "username": user["username"],
        "role":     user["role"],
    }

@app.get("/auth/has_users")
def has_users():
    """Returns whether any users exist. Used to detect first-time setup."""
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users;")
        count = cur.fetchone()[0]
    conn.close()
    return {"has_users": count > 0}

# USERS
@app.get("/users")
def list_users():
    conn = get_db()
    with dict_cursor(conn) as cur:
        cur.execute("SELECT id, username, role, created_at FROM users ORDER BY id;")
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/users", status_code=201)
def create_user(body: UserCreate):
    if body.role not in ("admin", "viewer"):
        raise HTTPException(status_code=422, detail="Role must be 'admin' or 'viewer'.")
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            password_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
            cur.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, %s) RETURNING id, username, role, created_at;
            """, (body.username.strip(), password_hash, body.role))
            row = cur.fetchone()
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=409, detail=f"Username '{body.username}' already exists.")
    conn.close()
    return dict(row)

@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int):
    conn = get_db()
    with conn.cursor() as cur:
        # Prevent deleting the last admin
        cur.execute("SELECT role FROM users WHERE id = %s;", (user_id,))
        user = cur.fetchone()
        if not user:
            conn.close()
            raise HTTPException(status_code=404, detail="User not found.")
        if user[0] == "admin":
            cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin';")
            if cur.fetchone()[0] <= 1:
                conn.close()
                raise HTTPException(status_code=400, detail="Cannot delete the last admin user.")
        cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
        conn.commit()
    conn.close()

# CAMERAS
@app.get("/cameras")
def list_cameras():
    conn = get_db()
    with dict_cursor(conn) as cur:
        cur.execute("SELECT * FROM cameras WHERE active = TRUE ORDER BY id;")
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/cameras", status_code=201)
def create_camera(body: CameraCreate):
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            INSERT INTO cameras (name, stream_url, location)
            VALUES (%s, %s, %s) RETURNING *;
        """, (body.name, body.stream_url, body.location))
        row = cur.fetchone()
        conn.commit()
    conn.close()
    return dict(row)

@app.put("/cameras/{camera_id}")
def update_camera(camera_id: int, body: CameraUpdate):
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM cameras WHERE id = %s;", (camera_id,))
        cam = cur.fetchone()
        if not cam:
            conn.close()
            raise HTTPException(status_code=404, detail="Camera not found")
        cur.execute("""
            UPDATE cameras SET
                name       = COALESCE(%s, name),
                stream_url = COALESCE(%s, stream_url),
                location   = COALESCE(%s, location),
                active     = COALESCE(%s, active)
            WHERE id = %s RETURNING *;
        """, (body.name, body.stream_url, body.location, body.active, camera_id))
        row = cur.fetchone()
        conn.commit()
    conn.close()
    return dict(row)

@app.delete("/cameras/{camera_id}", status_code=204)
def delete_camera(camera_id: int):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM cameras WHERE id = %s;", (camera_id,))
        conn.commit()
    conn.close()

# ZONES
@app.get("/zones")
def list_zones(camera_id: Optional[int] = None):
    conn = get_db()
    with dict_cursor(conn) as cur:
        sql = """
            SELECT z.*, c.name AS camera_name, COUNT(e.id) AS event_count
            FROM zones z
            LEFT JOIN cameras c ON z.camera_id = c.id
            LEFT JOIN events  e ON e.zone_id   = z.id
            WHERE 1=1
        """
        params = []
        if camera_id:
            sql += " AND z.camera_id = %s"; params.append(camera_id)
        sql += " GROUP BY z.id, c.name ORDER BY z.id"
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/zones", status_code=201)
def create_zone(body: ZoneCreate):
    if body.alert_level not in ("HIGH", "MEDIUM", "LOW"):
        raise HTTPException(status_code=422, detail="alert_level must be HIGH, MEDIUM or LOW")
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            INSERT INTO zones (name, alert_level, x1, y1, x2, y2, camera_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *;
        """, (body.name, body.alert_level, body.x1, body.y1, body.x2, body.y2, body.camera_id))
        row = cur.fetchone()
        conn.commit()
    conn.close()
    return dict(row)

@app.put("/zones/{zone_id}")
def update_zone(zone_id: int, body: ZoneUpdate):
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM zones WHERE id = %s;", (zone_id,))
        if not cur.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Zone not found")
        cur.execute("""
            UPDATE zones SET
                name        = COALESCE(%s, name),
                alert_level = COALESCE(%s, alert_level),
                x1          = COALESCE(%s, x1),
                y1          = COALESCE(%s, y1),
                x2          = COALESCE(%s, x2),
                y2          = COALESCE(%s, y2)
            WHERE id = %s RETURNING *;
        """, (body.name, body.alert_level, body.x1, body.y1, body.x2, body.y2, zone_id))
        row = cur.fetchone()
        conn.commit()
    conn.close()
    return dict(row)

@app.delete("/zones/{zone_id}", status_code=204)
def delete_zone(zone_id: int):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("UPDATE zones SET active = FALSE WHERE id = %s;", (zone_id,))
        conn.commit()
    conn.close()

@app.delete("/zones", status_code=204)
def clear_zones():
    """Soft-deletes all zones — preserves event history references."""
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("UPDATE zones SET active = FALSE;")
        conn.commit()
    conn.close()
# EVENTS
@app.get("/events")
def list_events(
    zone_id:        Optional[int] = None,
    detection_type: Optional[str] = None,
    alert_level:    Optional[str] = None,
    limit:          int = 200,
):
    conn = get_db()
    with dict_cursor(conn) as cur:
        sql = """
            SELECT e.id, e.detected_at, z.name AS zone,
                   e.detection_type, e.alert_level, e.snapshot_path
            FROM events e
            LEFT JOIN zones z ON e.zone_id = z.id
            WHERE 1=1
        """
        params = []
        if zone_id:
            sql += " AND e.zone_id = %s";        params.append(zone_id)
        if detection_type:
            sql += " AND e.detection_type = %s"; params.append(detection_type)
        if alert_level:
            sql += " AND e.alert_level = %s";    params.append(alert_level)
        sql += " ORDER BY e.detected_at DESC LIMIT %s"; params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/events", status_code=201)
def create_event(body: EventCreate):
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            INSERT INTO events (zone_id, detected_at, detection_type, alert_level, snapshot_path)
            VALUES (%s, NOW(), %s, %s, %s) RETURNING *;
        """, (body.zone_id, body.detection_type, body.alert_level, body.snapshot_path))
        row = cur.fetchone()
        conn.commit()
    conn.close()
    return dict(row)

@app.delete("/events", status_code=204)
def clear_events():
    """Deletes all events. Used by the dashboard Danger Zone button."""
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM events;")
        conn.commit()
    conn.close()

@app.delete("/events/{event_id}", status_code=204)
def delete_event(event_id: int):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM events WHERE id = %s;", (event_id,))
        conn.commit()
    conn.close()

# SUMMARY
@app.get("/summary")
def get_summary():
    today = datetime.now().date()
    conn  = get_db()
    with dict_cursor(conn) as cur:
        cur.execute("SELECT COUNT(*) AS c FROM events WHERE detected_at::date = %s;", (today,))
        events_today = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM events WHERE alert_level='HIGH' AND detected_at::date = %s;", (today,))
        high_alerts = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM zones;")
        zone_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM cameras WHERE active = TRUE;")
        camera_count = cur.fetchone()["c"]
    conn.close()
    return {
        "events_today": events_today,
        "high_alerts":  high_alerts,
        "zone_count":   zone_count,
        "camera_count": camera_count,
    }

# SEED DATA
@app.post("/seed", status_code=201)
def seed_data():
    """Populates sample zones and events. No-op if zones already exist."""
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM zones;")
        if cur.fetchone()[0] > 0:
            conn.close()
            return {"seeded": False, "message": "Database already has data."}

        sample_zones = [
            ("Front Door", "HIGH",   0.0, 0.0, 0.4, 0.5),
            ("Backyard",   "MEDIUM", 0.4, 0.0, 1.0, 0.6),
            ("Garage",     "LOW",    0.0, 0.5, 0.5, 1.0),
            ("Side Gate",  "HIGH",   0.5, 0.5, 1.0, 1.0),
        ]
        zone_ids = []
        for z in sample_zones:
            cur.execute("""
                INSERT INTO zones (name, alert_level, x1, y1, x2, y2)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
            """, z)
            zone_ids.append(cur.fetchone()[0])

        detection_types = ["Person Detected", "Motion Only"]
        alert_levels    = ["HIGH", "HIGH", "MEDIUM", "LOW", "LOW"]
        for _ in range(40):
            ts = datetime.now() - timedelta(minutes=random.randint(1, 60 * 72))
            cur.execute("""
                INSERT INTO events (zone_id, detected_at, detection_type, alert_level)
                VALUES (%s, %s, %s, %s);
            """, (
                random.choice(zone_ids), ts,
                random.choice(detection_types),
                random.choice(alert_levels),
            ))
        conn.commit()
    conn.close()
    return {"seeded": True, "message": "Sample data inserted successfully."}