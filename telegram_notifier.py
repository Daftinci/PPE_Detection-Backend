"""
Telegram violation notification module.

- Filters by confidence threshold (default 85%)
- Batches all violations within a time window into one message
- Sends annotated images as a media group
- Non-blocking: worker threads only call enqueue() (queue.put_nowait)
- Gracefully disabled if TELEGRAM_BOT_TOKEN is not set
"""

import asyncio
import json
import os
import queue
import threading
from datetime import datetime

import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from telegram import Bot, InputMediaPhoto
    from telegram.error import TelegramError
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False


SEVERITY_EMOJI = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🟢",
}


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        confidence_threshold: float = 0.85,
        batch_interval_seconds: int = 30,
    ):
        self._token = token
        self._chat_id = chat_id
        self._confidence_threshold = confidence_threshold
        self._batch_interval = batch_interval_seconds
        self._enabled = False
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._bot: "Bot | None" = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not _TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not installed — notifications disabled.")
            return
        if not self._token or not self._chat_id:
            logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — notifications disabled.")
            return
        self._enabled = True
        t = threading.Thread(target=self._sender_thread_main, daemon=True, name="telegram-sender")
        t.start()
        logger.info("Notifier started (threshold=%.2f, batch_interval=%ds)",
                    self._confidence_threshold, self._batch_interval)

    def enqueue(self, detection: dict) -> None:
        """Called by worker threads after a violation is saved. Non-blocking."""
        if not self._enabled:
            return
        try:
            if detection.get("confidence", 0) < self._confidence_threshold:
                return
            self._queue.put_nowait(detection)
        except queue.Full:
            logger.warning("Queue full — dropping notification for %s", detection.get("camera_location"))
        except Exception as exc:
            logger.error("enqueue error: %s", exc)

    # ------------------------------------------------------------------
    # Sender thread — owns its own asyncio event loop
    # ------------------------------------------------------------------

    def _sender_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._bot = Bot(token=self._token)
        try:
            loop.run_until_complete(self._schedule_batch_loop())
        except Exception as exc:
            logger.error("Sender thread crashed: %s", exc)

    async def _schedule_batch_loop(self) -> None:
        while True:
            await asyncio.sleep(self._batch_interval)
            try:
                await self._flush_batch()
            except Exception as exc:
                logger.error("Flush error: %s", exc)

    async def _flush_batch(self) -> None:
        # Drain all pending detections
        pending: list[dict] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not pending:
            return

        text = self._build_message(pending)
        image_paths = [
            d["image_path"]
            for d in pending
            if d.get("image_path") and os.path.isfile(d["image_path"])
        ]

        try:
            if not image_paths:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="HTML",
                )
            else:
                # Send text first so it appears above photos
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="HTML",
                )
                # Send images in chunks of 10 (Telegram limit)
                for chunk_start in range(0, len(image_paths), 10):
                    chunk = image_paths[chunk_start:chunk_start + 10]
                    media = []
                    for i, path in enumerate(chunk):
                        caption = self._image_caption(pending, path) if i == 0 else None
                        media.append(InputMediaPhoto(media=open(path, "rb"), caption=caption))
                    await self._bot.send_media_group(chat_id=self._chat_id, media=media)
                    if chunk_start + 10 < len(image_paths):
                        await asyncio.sleep(0.5)
        except TelegramError as exc:
            logger.error("Send failed: %s", exc)
        except Exception as exc:
            logger.error("Unexpected send error: %s", exc)

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def _build_message(self, detections: list[dict]) -> str:
        if len(detections) == 1:
            return self._format_single(detections[0])
        return self._format_batch(detections)

    def _format_single(self, d: dict) -> str:
        severity = d.get("severity", "unknown")
        emoji = SEVERITY_EMOJI.get(severity, "⚠️")
        violation_types = json.loads(d.get("violation_types", "[]"))
        missing_items = json.loads(d.get("missing_items", "[]"))
        ts = _parse_ts(d.get("timestamp", ""))
        conf = d.get("confidence", 0) * 100

        lines = [
            "🚨 <b>CCTV VIOLATION ALERT</b>",
            "",
            f"📍 Zone: {d.get('camera_zone', '-')}",
            f"📷 Camera: {d.get('camera_location', '-')}",
            f"🕐 Time: {ts}",
            f"{emoji} Severity: {severity.upper()}",
            f"🔍 Violations: {', '.join(violation_types) or '-'}",
        ]
        if missing_items:
            lines.append(f"🦺 Missing PPE: {', '.join(i.title() for i in missing_items)}")
        lines.append(f"📊 Confidence: {conf:.1f}%")
        return "\n".join(lines)

    def _format_batch(self, detections: list[dict]) -> str:
        n = len(detections)
        lines = [f"🚨 <b>CCTV VIOLATION ALERT — {n} violations detected</b>", ""]
        for i, d in enumerate(detections, 1):
            severity = d.get("severity", "unknown")
            emoji = SEVERITY_EMOJI.get(severity, "⚠️")
            violation_types = json.loads(d.get("violation_types", "[]"))
            ts = _parse_ts(d.get("timestamp", ""), time_only=True)
            conf = d.get("confidence", 0) * 100
            lines.append(
                f"{i}. <b>{d.get('camera_location', '-')}</b> ({d.get('camera_zone', '-')}) — {ts}"
            )
            lines.append(
                f"   {emoji} {severity.upper()} | {', '.join(violation_types) or '-'} | Conf: {conf:.1f}%"
            )
        return "\n".join(lines)

    def _image_caption(self, detections: list[dict], path: str) -> str:
        # Find the detection matching this image path
        for d in detections:
            if d.get("image_path") == path:
                severity = d.get("severity", "")
                violation_types = json.loads(d.get("violation_types", "[]"))
                conf = d.get("confidence", 0) * 100
                return f"{d.get('camera_location', '-')} | {severity.upper()} | {', '.join(violation_types)} | {conf:.1f}%"
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str, time_only: bool = False) -> str:
    try:
        dt = datetime.fromisoformat(ts_str)
        if time_only:
            return dt.strftime("%H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_notifier: "TelegramNotifier | None" = None


def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier(
            token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            confidence_threshold=float(os.getenv("TELEGRAM_CONFIDENCE_THRESHOLD", "0.85")),
            batch_interval_seconds=int(os.getenv("TELEGRAM_BATCH_INTERVAL_SECONDS", "30")),
        )
    return _notifier
