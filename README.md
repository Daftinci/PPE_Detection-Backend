# CCTVAi PPE Detection — Backend

FastAPI backend for real-time PPE (Personal Protective Equipment) and fire/smoke detection via CCTV cameras using YOLO models.

## Features

- Real-time PPE detection (helmet, coverall) via RTSP camera streams
- Fire and smoke detection running in parallel
- MJPEG live stream endpoint per camera
- JWT authentication
- Telegram alert batching
- SQLite detection history with CSV export
- Per-zone compliance reporting

## Requirements

- Python 3.10+
- YOLO model weights (`DevSynapse-hybrid.pt`, `firedetect-11x.pt`) — not included in repo

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your values

# Add cameras to data.json (see Camera Config section below)

# Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `PPE_MODEL_PATH` | `DevSynapse-hybrid.pt` | Path to PPE YOLO model |
| `FIRE_MODEL_PATH` | `firedetect-11x.pt` | Path to fire detection YOLO model |
| `IMAGE_RETENTION_DAYS` | `30` | Days to keep violation images and DB records (0 = forever) |
| `SAVE_RAW_FRAMES` | `false` | Save unannotated frames alongside violations |
| `TELEGRAM_BOT_TOKEN` | — | Bot token from @BotFather; leave unset to disable |
| `TELEGRAM_CHAT_ID` | — | Target chat/group ID |
| `TELEGRAM_CONFIDENCE_THRESHOLD` | `0.85` | Min confidence to trigger a Telegram notification |
| `TELEGRAM_BATCH_INTERVAL_SECONDS` | `30` | How often to flush the notification queue |

## Camera Config (`data.json`)

```json
[
  {
    "name": "Gate A",
    "url": "rtsp://user:pass@192.168.1.10:554/stream",
    "path": "field/Zone1/Gate A"
  }
]
```

The `path` field determines the zone — the second path segment (e.g. `Zone1`) is extracted automatically.

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Service health check |
| `POST` | `/api/v1/auth/login` | — | Returns JWT token |
| `GET` | `/api/v1/cameras` | Bearer | List all cameras with live state |
| `GET` | `/api/v1/cameras/{id}` | Bearer | Single camera details |
| `GET` | `/api/v1/cameras/{id}/stream` | `?token=` | MJPEG live stream |
| `GET` | `/api/v1/detections` | Bearer | Detection history (`?limit=`, `?offset=`) |
| `GET` | `/api/v1/alerts` | Bearer | Unacknowledged detections |
| `POST` | `/api/v1/alerts/{id}/acknowledge` | Bearer | Mark alert as acknowledged |
| `GET` | `/api/v1/dashboard/stats` | Bearer | Counts and compliance rate |
| `GET` | `/api/v1/zones` | Bearer | Per-zone compliance |
| `GET` | `/api/v1/reports/summary` | Bearer | Aggregated report (`?period=7d\|30d\|90d`) |
| `GET` | `/api/v1/reports/export` | Bearer | CSV download |
| `DELETE` | `/api/v1/admin/detections` | Bearer | Delete all detection records and images |
| `DELETE` | `/api/v1/admin/cleanup` | Bearer | Trigger manual retention cleanup |

The MJPEG stream uses `?token=<jwt>` instead of `Authorization:` header to support direct use in `<img src>`.

## Detection Models

| File | Purpose | Confidence Threshold |
|---|---|---|
| `DevSynapse-hybrid.pt` | PPE — helmet / coverall | 0.81 |
| `firedetect-11x.pt` | Fire / smoke | 0.40 |

## Architecture

```
main.py              FastAPI app, REST endpoints
auth.py              JWT auth (python-jose)
db.py                SQLite helpers (detections.db)
worker.py            Background detection threads, shared camera_states dict
telegram_notifier.py Async Telegram alert batching
model.py             YOLO inference wrappers
data.json            Camera config
field/               Annotated violation images → served at /images/*
```

Each camera spawns two daemon threads: a capture loop (frame reader) and an inference loop. They share a `Queue(maxsize=1)` so stale frames are evicted automatically. Both models are loaded once per inference thread and reused.

- PPE violations require a 750 ms grace period before saving
- Fire detections re-save every 10 s while the hazard remains active
- The MJPEG stream reads JPEG bytes from `camera_states` — no second VideoCapture connection
