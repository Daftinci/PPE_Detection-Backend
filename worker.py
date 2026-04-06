"""
Detection background worker.
Each camera from data.json runs in its own pair of daemon threads:

  - _capture_loop  : reads frames from VideoCapture as fast as possible,
                     puts the latest frame into a Queue(maxsize=1).
                     Handles connection / reconnection and updates online/offline status.

  - _run_camera    : inference thread.
                     Pulls frames from the queue (blocks until one is ready),
                     runs PPE every frame and fire every FIRE_INTERVAL frames,
                     then stores the annotated JPEG in camera_states so the
                     MJPEG stream endpoint can serve it without touching the camera.
"""

import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime

import cv2
import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

from db import insert_detection
from telegram_notifier import get_notifier

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

FIRE_INTERVAL = 5   # run fire model every N inference frames

# ---------------------------------------------------------------------------
# Shared state (read by main.py to build /cameras responses)
# ---------------------------------------------------------------------------

camera_states: dict[str, dict] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cameras() -> list[dict]:
    path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(path) as f:
        return json.load(f)


def camera_id_from(camera: dict) -> str:
    return camera["name"].replace(" ", "_").lower()


def zone_from(camera: dict) -> str:
    """Extract zone name from the path field: field/<zone>/..."""
    parts = camera.get("path", "").split("/")
    return parts[1] if len(parts) > 1 else "Unknown"


def _severity(violation_labels: set) -> str:
    fire_labels = violation_labels - {"No Helmet", "No Coverall"}
    if fire_labels:
        return "critical"
    if "No Helmet" in violation_labels and "No Coverall" in violation_labels:
        return "high"
    if "No Helmet" in violation_labels:
        return "medium"
    return "low"


def _missing_items(violation_labels: set) -> list[str]:
    items = []
    if "No Helmet" in violation_labels:
        items.append("helmet")
    if "No Coverall" in violation_labels:
        items.append("vest")
    return items


THUMBNAIL_COLORS = ["#1e40af", "#15803d", "#b91c1c", "#a16207", "#6d28d9", "#0e7490"]


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------

def _capture_loop(camera: dict, cam_id: str, frame_queue: "queue.Queue") -> None:
    """
    Fast loop: reads raw frames and drops them into frame_queue.
    Queue is maxsize=1 — if inference is slower than capture, stale frames
    are evicted so inference always gets the freshest frame available.
    """
    def _set_state(**kwargs):
        with _lock:
            camera_states[cam_id].update(kwargs)

    while True:  # reconnect loop
        cap = cv2.VideoCapture(camera["url"])
        if not cap.isOpened():
            logger.warning("Cannot open camera: %s — retrying in 10 s", camera["name"])
            _set_state(status="offline")
            time.sleep(10)
            continue

        _set_state(status="online", fps=cap.get(cv2.CAP_PROP_FPS))
        os.makedirs(camera.get("path", ""), exist_ok=True)
        logger.info("Capture started: %s", camera["name"])

        while True:
            ret, frame = cap.read()
            if not ret:
                _set_state(status="offline")
                break

            _set_state(last_seen=datetime.now().isoformat())

            # Evict the stale frame before inserting the new one
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(frame)

        cap.release()


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

def _run_camera(camera: dict, color: str) -> None:
    cam_id = camera_id_from(camera)
    zone   = zone_from(camera)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load both models once
    ppe_model_path  = os.getenv("PPE_MODEL_PATH",  "DevSynapse-hybrid.pt")
    fire_model_path = os.getenv("FIRE_MODEL_PATH", "firedetect-11x.pt")
    try:
        model      = YOLO(ppe_model_path).to(device)
        fire_model = YOLO(fire_model_path).to(device)
    except Exception as exc:
        logger.critical("Failed to load models (PPE: %s, Fire: %s): %s", ppe_model_path, fire_model_path, exc)
        raise

    def _set_state(**kwargs):
        with _lock:
            camera_states[cam_id].update(kwargs)

    # Initialise shared state
    with _lock:
        camera_states[cam_id] = {
            "id":              cam_id,
            "location":        camera["name"],
            "zone":            zone,
            "status":          "offline",
            "fps":             0.0,
            "resolution":      "1280x720",
            "last_seen":       datetime.now().isoformat(),
            "detection_count": 0,
            "thumbnail_color": color,
            "url":             camera["url"],
            "path":            camera.get("path", ""),
        }

    # Start capture thread
    frame_queue: "queue.Queue" = queue.Queue(maxsize=1)
    cap_thread = threading.Thread(
        target=_capture_loop, args=(camera, cam_id, frame_queue), daemon=True
    )
    cap_thread.start()
    logger.info("Inference ready: %s on %s", camera["name"], device.upper())

    frame_count      = 0
    last_fire_dets:  list = []
    last_fire_names: dict = {}

    grace_period       = 0.75   # seconds a PPE violation must persist before saving
    viol_start         = None
    incident_saved     = False  # True once saved; resets only when PPE violation clears
    FIRE_SAVE_INTERVAL = 10.0   # re-save fire every N seconds while still detected
    last_fire_save     = 0.0

    while True:
        # Block until a frame is available; timeout lets us survive a camera dropout
        try:
            frame = frame_queue.get(timeout=5.0)
        except queue.Empty:
            continue

        raw_frame    = frame.copy()
        frame_count += 1

        # --- PPE inference (every frame) ---
        ppe_results = model(frame, verbose=False)
        ppe_dets    = ppe_results[0].boxes.data.cpu().numpy()
        ppe_names   = ppe_results[0].names

        # --- Fire inference (every FIRE_INTERVAL frames) ---
        if frame_count % FIRE_INTERVAL == 0:
            fire_results     = fire_model(frame, verbose=False)
            last_fire_dets   = fire_results[0].boxes.data.cpu().numpy()
            last_fire_names  = fire_results[0].names

        fire_dets  = last_fire_dets
        fire_names = last_fire_names

        save_viol    = False
        viol_labels: set[str] = set()
        max_conf     = 0.0

        for x1, y1, x2, y2, conf, cls in ppe_dets:
            cls = int(cls)
            if conf > 0.81:
                name      = ppe_names[cls]
                color_box = (0, 255, 0)
                if name == "Non Helmet":
                    color_box = (0, 0, 255)
                    save_viol = True
                    viol_labels.add("No Helmet")
                    max_conf = max(max_conf, float(conf))
                elif name == "Non Coverall":
                    color_box = (0, 0, 255)
                    save_viol = True
                    viol_labels.add("No Coverall")
                    max_conf = max(max_conf, float(conf))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        for x1, y1, x2, y2, conf, cls in fire_dets:
            cls = int(cls)
            if conf > 0.40:
                name = fire_names[cls]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 69, 255), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 69, 255), 2)
                save_viol = True
                viol_labels.add(name)
                max_conf = max(max_conf, float(conf))

        # Push annotated JPEG to camera_states for the stream endpoint
        _, _buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _lock:
            camera_states[cam_id]["latest_frame"] = _buf.tobytes()

        now = time.time()
        ts  = datetime.now()

        if save_viol:
            _set_state(status="alert")
            is_fire = bool(viol_labels - {"No Helmet", "No Coverall"})

            if is_fire:
                # Fire: save every FIRE_SAVE_INTERVAL seconds while fire is present
                if now - last_fire_save >= FIRE_SAVE_INTERVAL:
                    _save_violation(camera, cam_id, zone, frame, raw_frame,
                                    ts, viol_labels, max_conf)
                    last_fire_save = now
            else:
                # PPE: save once per continuous incident (reset when person leaves)
                if viol_start is None:
                    viol_start     = now
                    incident_saved = False
                elif not incident_saved and now - viol_start >= grace_period:
                    _save_violation(camera, cam_id, zone, frame, raw_frame,
                                    ts, viol_labels, max_conf)
                    incident_saved = True
        else:
            with _lock:
                if camera_states[cam_id]["status"] == "alert":
                    camera_states[cam_id]["status"] = "online"
            viol_start     = None
            incident_saved = False


# ---------------------------------------------------------------------------
# Violation persistence
# ---------------------------------------------------------------------------

def _save_violation(camera: dict, cam_id: str, zone: str,
                    frame, raw_frame,
                    ts: datetime, viol_labels: set, max_conf: float) -> None:
    ts_str = ts.strftime("%Y-%m-%d_%H;%M;%S")
    ppe_l  = viol_labels & {"No Helmet", "No Coverall"}
    fire_l = viol_labels - {"No Helmet", "No Coverall"}

    if ppe_l == {"No Helmet", "No Coverall"}:
        subfolder = "No Helmet & No Coverall"
    elif "No Helmet" in ppe_l:
        subfolder = "No Helmet"
    elif "No Coverall" in ppe_l:
        subfolder = "No Coverall"
    elif fire_l:
        subfolder = "Fire"
    else:
        subfolder = "Other"

    vfolder = os.path.join(camera.get("path", ""), subfolder)
    os.makedirs(vfolder, exist_ok=True)

    img_path = os.path.join(vfolder, f"{camera['name']}_{ts_str}.jpg")
    cv2.imwrite(img_path, frame)

    if os.getenv("SAVE_RAW_FRAMES", "").lower() in ("1", "true", "yes"):
        raw_dir = os.getenv("RAW_FRAMES_DIR", "dummy")
        os.makedirs(raw_dir, exist_ok=True)
        cv2.imwrite(os.path.join(raw_dir, f"{camera['name']}_{ts_str}.jpg"), raw_frame)

    detection_id = str(uuid.uuid4())
    insert_detection({
        "id":              detection_id,
        "camera_id":       cam_id,
        "camera_location": camera["name"],
        "camera_zone":     zone,
        "timestamp":       ts.isoformat(),
        "severity":        _severity(viol_labels),
        "ppe_status":      "violation",
        "missing_items":   json.dumps(_missing_items(viol_labels)),
        "violation_types": json.dumps(list(viol_labels)),
        "confidence":      round(max_conf, 3),
        "acknowledged":    0,
        "image_path":      img_path,
    })

    with _lock:
        camera_states[cam_id]["detection_count"] += 1

    logger.info("Violation saved: %s at %s -> %s", ", ".join(viol_labels), camera["name"], img_path)

    get_notifier().enqueue({
        "id":              detection_id,
        "camera_id":       cam_id,
        "camera_location": camera["name"],
        "camera_zone":     zone,
        "timestamp":       ts.isoformat(),
        "severity":        _severity(viol_labels),
        "missing_items":   json.dumps(_missing_items(viol_labels)),
        "violation_types": json.dumps(list(viol_labels)),
        "confidence":      round(max_conf, 3),
        "image_path":      img_path,
    })


# ---------------------------------------------------------------------------
# Entry point called by main.py on startup
# ---------------------------------------------------------------------------

def start_workers() -> None:
    cameras = load_cameras()
    for i, cam in enumerate(cameras):
        color = THUMBNAIL_COLORS[i % len(THUMBNAIL_COLORS)]
        t = threading.Thread(target=_run_camera, args=(cam, color), daemon=True)
        t.start()
