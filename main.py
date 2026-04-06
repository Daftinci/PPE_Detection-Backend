"""
CCTVAi PPE Detection — FastAPI Backend
Run with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import os

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import csv
import io
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import authenticate, create_token, verify_token, verify_token_query
from db import (
    DB_PATH,
    acknowledge_detection,
    cleanup_old_images,
    delete_all_detections,
    get_detections,
    get_report_data,
    get_stats,
    get_zone_stats,
    init_db,
)
from telegram_notifier import get_notifier
from worker import (
    THUMBNAIL_COLORS,
    _lock,
    camera_id_from,
    camera_states,
    load_cameras,
    start_workers,
    zone_from,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    retention_days = int(os.getenv("IMAGE_RETENTION_DAYS", "30"))
    if retention_days > 0:
        result = cleanup_old_images(retention_days)
        if result["deleted_records"] > 0:
            logger.info(
                "Startup cleanup: removed %d old records, %d files",
                result["deleted_records"], result["deleted_files"],
            )
    get_notifier().start()
    start_workers()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")


app = FastAPI(title="CCTVAi PPE Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve violation images as static files under /images
_field_dir = os.path.join(os.path.dirname(__file__), "field")
if os.path.isdir(_field_dir):
    app.mount("/images", StaticFiles(directory=_field_dir), name="images")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
    with _lock:
        workers_up = len(camera_states) > 0
    return {
        "status":         "ok" if db_ok else "degraded",
        "db":             "ok" if db_ok else "error",
        "workers_started": workers_up,
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/auth/login")
def login(req: LoginRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": create_token(req.username)}


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def _camera_response(cam: dict, index: int) -> dict:
    cam_id = camera_id_from(cam)
    with _lock:
        state = dict(camera_states.get(cam_id, {}))
    return {
        "id":             cam_id,
        "location":       cam["name"],
        "zone":           zone_from(cam),
        "status":         state.get("status", "offline"),
        "fps":            round(state.get("fps", 0.0), 1),
        "resolution":     "1280x720",
        "lastSeen":       state.get("last_seen", ""),
        "detectionCount": state.get("detection_count", 0),
        "thumbnailColor": THUMBNAIL_COLORS[index % len(THUMBNAIL_COLORS)],
    }


@app.get("/api/v1/cameras")
def get_cameras(user: str = Depends(verify_token)):
    cameras = load_cameras()
    return [_camera_response(cam, i) for i, cam in enumerate(cameras)]


@app.get("/api/v1/cameras/{camera_id}")
def get_camera(camera_id: str, user: str = Depends(verify_token)):
    for i, cam in enumerate(load_cameras()):
        if camera_id_from(cam) == camera_id:
            return _camera_response(cam, i)
    raise HTTPException(status_code=404, detail="Camera not found")


# ---------------------------------------------------------------------------
# MJPEG live stream  (token accepted as ?token= query param for <img src>)
# ---------------------------------------------------------------------------

def _frame_generator(cam_id: str):
    """Yield MJPEG frames from the worker's shared annotated-frame buffer.

    The worker thread owns the VideoCapture exclusively; this generator just
    reads the latest encoded JPEG it deposited in camera_states, so there is
    no second connection to the camera and detection never stops.
    """
    try:
        while True:
            with _lock:
                frame_bytes = camera_states.get(cam_id, {}).get("latest_frame")
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps ceiling for the stream
    except GeneratorExit:
        logger.info("Stream client disconnected: %s", cam_id)


@app.get("/api/v1/cameras/{camera_id}/stream")
def stream_camera(camera_id: str, user: str = Depends(verify_token_query)):
    with _lock:
        known = camera_id in camera_states
    if not known:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def _detection_row(row) -> dict:
    return {
        "id":             row["id"],
        "cameraId":       row["camera_id"],
        "cameraLocation": row["camera_location"],
        "timestamp":      row["timestamp"],
        "severity":       row["severity"],
        "ppeStatus":      row["ppe_status"],
        "missingItems":   json.loads(row["missing_items"]),
        "violationTypes": json.loads(row["violation_types"]),
        "confidence":     row["confidence"],
        "acknowledged":   bool(row["acknowledged"]),
    }


@app.get("/api/v1/detections")
def list_detections(
    limit: Optional[int] = Query(None),
    offset: int = Query(0, ge=0),
    user: str = Depends(verify_token),
):
    return [_detection_row(r) for r in get_detections(limit, offset)]


# ---------------------------------------------------------------------------
# Alerts  (unacknowledged detections surfaced as Alert objects)
# ---------------------------------------------------------------------------

def _alert_message(violation_types: list[str], missing_items: list[str]) -> str:
    parts = []
    if "No Helmet" in violation_types:
        parts.append("no helmet")
    if "No Coverall" in violation_types:
        parts.append("no coverall")
    fire = [v for v in violation_types if v not in {"No Helmet", "No Coverall"}]
    if fire:
        parts.append("fire / smoke detected")
    return "PPE violation: " + ", ".join(parts) if parts else "Violation detected"


@app.get("/api/v1/alerts")
def list_alerts(user: str = Depends(verify_token)):
    result = []
    for row in get_detections():
        if row["acknowledged"]:
            continue
        viol_types   = json.loads(row["violation_types"])
        missing      = json.loads(row["missing_items"])
        result.append({
            "id":           row["id"],
            "detectionId":  row["id"],
            "cameraId":     row["camera_id"],
            "location":     row["camera_location"],
            "timestamp":    row["timestamp"],
            "severity":     row["severity"],
            "message":      _alert_message(viol_types, missing),
            "acknowledged": False,
        })
    return result


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, user: str = Depends(verify_token)):
    acknowledge_detection(alert_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

@app.get("/api/v1/dashboard/stats")
def dashboard_stats(user: str = Depends(verify_token)):
    cameras = load_cameras()
    stats   = get_stats()

    total   = len(cameras)
    active  = 0
    alert_c = 0
    for cam in cameras:
        with _lock:
            s = camera_states.get(camera_id_from(cam), {}).get("status", "offline")
        if s in ("online", "alert"):
            active += 1
        if s == "alert":
            alert_c += 1

    offline  = total - active
    today_d  = stats["today_detections"]
    today_v  = stats["today_violations"]
    compliance = round((1 - today_v / today_d) * 100, 1) if today_d > 0 else 100.0

    return {
        "totalCameras":    total,
        "activeCameras":   active,
        "offlineCameras":  offline,
        "alertCameras":    alert_c,
        "todayDetections": today_d,
        "todayViolations": today_v,
        "complianceRate":  compliance,
        "avgResponseTime": None,
        "activeAlerts":    stats["active_alerts"],
    }


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

@app.get("/api/v1/zones")
def list_zones(user: str = Depends(verify_token)):
    cameras = load_cameras()

    zone_cams: dict[str, list] = {}
    for cam in cameras:
        z = zone_from(cam)
        zone_cams.setdefault(z, []).append(cam)

    zone_db = {row["camera_zone"]: row for row in get_zone_stats()}

    result = []
    for i, (zone, cams) in enumerate(zone_cams.items()):
        zs = zone_db.get(zone)
        if zs and zs["total"] > 0:
            compliance = round((1 - zs["violations"] / zs["total"]) * 100, 1)
        else:
            compliance = 100.0
        result.append({
            "id":             f"zone_{i}",
            "name":           zone,
            "cameraCount":    len(cams),
            "complianceRate": compliance,
        })
    return result


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

_PERIOD_DAYS = {"7d": 7, "30d": 30, "90d": 90}


@app.get("/api/v1/reports/summary")
def report_summary(
    period: str = Query("30d"),
    user: str = Depends(verify_token),
):
    days = _PERIOD_DAYS.get(period, 30)
    rows = get_report_data(days)

    total      = len(rows)
    violations = sum(1 for r in rows if r["ppe_status"] == "violation")
    compliance = round((1 - violations / total) * 100, 1) if total > 0 else 100.0

    # --- by day ---
    by_day: dict[str, dict] = {}
    for r in rows:
        d = r["timestamp"][:10]
        if d not in by_day:
            by_day[d] = {"date": d, "violations": 0, "compliant": 0}
        if r["ppe_status"] == "violation":
            by_day[d]["violations"] += 1
        else:
            by_day[d]["compliant"] += 1

    all_days = []
    for i in range(days):
        d = (date.today() - timedelta(days=days - 1 - i)).isoformat()
        all_days.append(by_day.get(d, {"date": d, "violations": 0, "compliant": 0}))

    # --- by zone ---
    by_zone: dict[str, dict] = {}
    for r in rows:
        z = r["camera_zone"]
        if z not in by_zone:
            by_zone[z] = {"zone": z, "violations": 0, "total": 0}
        by_zone[z]["total"] += 1
        if r["ppe_status"] == "violation":
            by_zone[z]["violations"] += 1

    zone_list = [
        {
            "zone":       z["zone"],
            "violations": z["violations"],
            "compliance": round((1 - z["violations"] / z["total"]) * 100, 1) if z["total"] > 0 else 100.0,
        }
        for z in by_zone.values()
    ]

    # --- by PPE item ---
    item_counts: dict[str, int] = {}
    for r in rows:
        for item in json.loads(r["missing_items"]):
            item_counts[item] = item_counts.get(item, 0) + 1

    by_ppe = [
        {
            "item":       item,
            "violations": count,
            "percentage": round(count / violations * 100, 1) if violations > 0 else 0.0,
        }
        for item, count in item_counts.items()
    ]

    top = max(item_counts, key=item_counts.get) if item_counts else "helmet"

    return {
        "period":           period,
        "totalInspections": total,
        "violations":       violations,
        "compliance":       compliance,
        "topViolationType": top,
        "byDay":            all_days,
        "byZone":           zone_list,
        "byPPEItem":        by_ppe,
    }

@app.delete("/api/v1/admin/detections")
def delete_all(user: str = Depends(verify_token)):
    result = delete_all_detections()
    logger.info("Delete all: removed %d records, %d files", result["deleted_records"], result["deleted_files"])
    return result


@app.delete("/api/v1/admin/cleanup")
def trigger_cleanup(user: str = Depends(verify_token)):
    days = int(os.getenv("IMAGE_RETENTION_DAYS", "1"))
    if days == 0:
        return {"message": "Retention disabled (IMAGE_RETENTION_DAYS=0)", "deleted_records": 0, "deleted_files": 0}
    result = cleanup_old_images(days)
    logger.info("Manual cleanup: removed %d records, %d files", result["deleted_records"], result["deleted_files"])
    return result


@app.get("/api/v1/reports/export")
def export_report_csv(
    period: str = Query("30d"),
    user: str = Depends(verify_token),
):
    days = _PERIOD_DAYS.get(period, 30)
    rows = get_report_data(days)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "ID", "Timestamp", "Camera", "Zone",
        "Severity", "PPE Status", "Missing Items",
        "Violation Types", "Confidence", "Acknowledged",
    ])
    for r in rows:
        writer.writerow([
            r["id"],
            r["timestamp"],
            r["camera_location"],
            r["camera_zone"],
            r["severity"],
            r["ppe_status"],
            ", ".join(json.loads(r["missing_items"])),
            ", ".join(json.loads(r["violation_types"])),
            r["confidence"],
            "Yes" if r["acknowledged"] else "No",
        ])

    filename = f"ppe_report_{period}_{date.today().isoformat()}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
