"""SQLite helpers — every function opens its own connection (thread-safe)."""

import json
import logging
import sqlite3
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "detections.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id              TEXT PRIMARY KEY,
            camera_id       TEXT NOT NULL,
            camera_location TEXT NOT NULL,
            camera_zone     TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            severity        TEXT NOT NULL,
            ppe_status      TEXT NOT NULL,
            missing_items   TEXT NOT NULL,   -- JSON array of PPEItem strings
            violation_types TEXT NOT NULL,   -- JSON array of raw label strings
            confidence      REAL NOT NULL,
            acknowledged    INTEGER NOT NULL DEFAULT 0,
            image_path      TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_det_timestamp    ON detections(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_det_camera_id    ON detections(camera_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_det_acknowledged ON detections(acknowledged)")
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def insert_detection(row: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR IGNORE INTO detections
            (id, camera_id, camera_location, camera_zone, timestamp,
             severity, ppe_status, missing_items, violation_types,
             confidence, acknowledged, image_path)
        VALUES
            (:id, :camera_id, :camera_location, :camera_zone, :timestamp,
             :severity, :ppe_status, :missing_items, :violation_types,
             :confidence, :acknowledged, :image_path)
    """, row)
    conn.commit()
    conn.close()


def acknowledge_detection(detection_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE detections SET acknowledged = 1 WHERE id = ?", (detection_id,)
    )
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    conn.row_factory = sqlite3.Row
    return conn.execute(sql, params).fetchall()


def get_detections(limit: int | None = None, offset: int = 0) -> list:
    conn = sqlite3.connect(DB_PATH)
    if limit:
        rows = _rows(conn, "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
    else:
        rows = _rows(conn, "SELECT * FROM detections ORDER BY timestamp DESC LIMIT -1 OFFSET ?", (offset,))
    conn.close()
    return rows


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    today    = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    def scalar(sql: str, params: tuple = ()) -> int:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchone()[0]

    today_det  = scalar(
        "SELECT COUNT(*) FROM detections WHERE timestamp >= ? AND timestamp < ?",
        (today, tomorrow)
    )
    today_viol = scalar(
        "SELECT COUNT(*) FROM detections WHERE timestamp >= ? AND timestamp < ? AND ppe_status = 'violation'",
        (today, tomorrow)
    )
    active_alerts = scalar("SELECT COUNT(*) FROM detections WHERE acknowledged = 0")
    conn.close()
    return {
        "today_detections": today_det,
        "today_violations": today_viol,
        "active_alerts":    active_alerts,
    }


def get_zone_stats() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = _rows(conn, """
        SELECT
            camera_zone,
            COUNT(*) AS total,
            SUM(CASE WHEN ppe_status = 'violation' THEN 1 ELSE 0 END) AS violations
        FROM detections
        GROUP BY camera_zone
    """)
    conn.close()
    return rows


def get_report_data(period_days: int = 30) -> list:
    conn = sqlite3.connect(DB_PATH)
    since = (date.today() - timedelta(days=period_days)).isoformat()
    rows = _rows(conn, "SELECT * FROM detections WHERE timestamp >= ? ORDER BY timestamp DESC", (since,))
    conn.close()
    return rows


def delete_all_detections() -> dict:
    """Delete every detection record and its associated image file.

    Returns {"deleted_records": int, "deleted_files": int, "errors": int}.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, image_path FROM detections").fetchall()

    deleted_files = 0
    errors = 0
    for row in rows:
        if row["image_path"] and os.path.isfile(row["image_path"]):
            try:
                os.remove(row["image_path"])
                deleted_files += 1
            except OSError as exc:
                logger.warning("Could not delete image %s: %s", row["image_path"], exc)
                errors += 1

    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return {"deleted_records": len(rows), "deleted_files": deleted_files, "errors": errors}


def cleanup_old_images(retention_days: int) -> dict:
    """Delete image files and DB records older than retention_days.

    Returns {"deleted_records": int, "deleted_files": int, "errors": int}.
    """
    cutoff = (date.today() - timedelta(days=retention_days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, image_path FROM detections WHERE timestamp < ?", (cutoff,)
    ).fetchall()

    deleted_files = 0
    errors = 0
    ids_to_delete = []
    for row in rows:
        ids_to_delete.append(row["id"])
        if row["image_path"] and os.path.isfile(row["image_path"]):
            try:
                os.remove(row["image_path"])
                deleted_files += 1
            except OSError as exc:
                logger.warning("Could not delete image %s: %s", row["image_path"], exc)
                errors += 1

    if ids_to_delete:
        placeholders = ",".join("?" * len(ids_to_delete))
        conn.execute(f"DELETE FROM detections WHERE id IN ({placeholders})", ids_to_delete)
        conn.commit()

    conn.close()
    return {"deleted_records": len(ids_to_delete), "deleted_files": deleted_files, "errors": errors}
