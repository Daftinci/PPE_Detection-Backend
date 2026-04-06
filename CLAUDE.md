# CCTVAi PPE Detection — Backend

FastAPI backend for real-time PPE (Personal Protective Equipment) and fire detection via CCTV cameras.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Architecture

```
main.py              FastAPI app, all REST endpoints
auth.py              JWT auth (python-jose), hardcoded user store
db.py                SQLite helpers (detections.db), one connection per call
worker.py            Background detection threads, shared camera_states dict
telegram_notifier.py Async Telegram alert batching (non-blocking enqueue)
data.json            Camera config: [{name, url, path}]
field/               Annotated violation images, served as /images/* static files
dummy/               Raw (unannotated) frames saved alongside violations
```

## Key design points

- Each camera spawns two daemon threads: `_capture_loop` (frame reader) and `_run_camera` (inference). They share a `Queue(maxsize=1)` — stale frames are evicted automatically.
- Both YOLO models (`DevSynapse-hybrid.pt` for PPE, `firedetect-11x.pt` for fire) are loaded once per inference thread and reused.
- PPE violations require a 750 ms grace period before saving; fire re-saves every 10 s while active.
- `camera_states` is a shared dict protected by `_lock`. The MJPEG stream endpoint reads JPEG bytes from it — no second VideoCapture connection.
- MJPEG stream auth uses `?token=` query param (needed for `<img src>`); all other endpoints use `Authorization: Bearer`.

## Models

| File | Purpose | Confidence threshold |
|------|---------|----------------------|
| `DevSynapse-hybrid.pt` | PPE (helmet/coverall) | 0.81 |
| `firedetect-11x.pt` | Fire / smoke | 0.40 |

## Environment variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | — | Bot token; leave unset to disable Telegram |
| `TELEGRAM_CHAT_ID` | — | Target chat / group ID |
| `TELEGRAM_CONFIDENCE_THRESHOLD` | `0.85` | Min confidence to send a notification |
| `TELEGRAM_BATCH_INTERVAL_SECONDS` | `30` | How often to flush the notification queue |

## Endpoints

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| POST | `/api/v1/auth/login` | — | Returns JWT |
| GET | `/api/v1/cameras` | Bearer | List all cameras + live state |
| GET | `/api/v1/cameras/{id}` | Bearer | Single camera |
| GET | `/api/v1/cameras/{id}/stream` | `?token=` | MJPEG live stream |
| GET | `/api/v1/detections` | Bearer | All detections, optional `?limit=` |
| GET | `/api/v1/alerts` | Bearer | Unacknowledged detections |
| POST | `/api/v1/alerts/{id}/acknowledge` | Bearer | Mark alert as read |
| GET | `/api/v1/dashboard/stats` | Bearer | Counts and compliance rate |
| GET | `/api/v1/zones` | Bearer | Per-zone compliance |
| GET | `/api/v1/reports/summary` | Bearer | Aggregated report (`?period=7d\|30d\|90d`) |
| GET | `/api/v1/reports/export` | Bearer | CSV download |

## Camera config (data.json)

```json
[
  {
    "name": "Gate A",
    "url": "rtsp://...",
    "path": "field/Zone1/Gate A"
  }
]
```

`zone_from()` extracts the zone from `path` (second path segment).
