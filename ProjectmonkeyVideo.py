"""
monkey_detector_flask.py — NoIR V2 Camera + YOLO Detection + Web UI
════════════════════════════════════════════════════════════════════════════
Runs under system Python (picamera2 is system-installed).
Detects monkeys visually and sends trigger to alert_service via Unix socket.

Run:
    python3 monkey_detector_flask.py

systemd service example:
    ExecStart=/usr/bin/python3 \
              /home/monkeyaudio/monkey/video/monkey_detector_flask.py
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
import threading
import struct
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

# alert_client lives next to this file — stdlib only, no venv needed
from alert_client import trigger_alert

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MonkeyDetector")
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & DEFAULTS
# ══════════════════════════════════════════════════════════════════════

CALIB_FILE         = "camera_calib.json"
MODEL_PATH_DEFAULT = "MONKEY_DETECTION_YOLOV8(N).pt"

CAM_WIDTH,  CAM_HEIGHT = 1280, 720
DISPLAY_SCALE          = 0.75
JPEG_QUALITY           = 75

YOLO_IMG_SIZE       = 128
YOLO_CONF_THRESHOLD = 0.65
YOLO_DEVICE         = "cpu"
YOLO_USE_FP16       = False

BLUE_CORRECTION     = 1.15
IR_ATTENUATION      = 0.50
HISTOGRAM_SMOOTHING = 0.1

ALERT_COOLDOWN_SECONDS = 3.0

SHM_FRAME_SIZE = CAM_WIDTH * CAM_HEIGHT * 3 + 4
SHM_STATS_SIZE = 256
SHM_JPEG_NAME  = "monkey_jpeg_shm"
SHM_STATS_NAME = "monkey_stats_shm"

COLORS = {
    "Human":      (80,  80,  255),
    "Monkey":     (0,  220,   0),
    "Monkey_Warn":(0,    0,  255),
    "Other":      (200, 200, 200),
    "HUD_Text":   (0,  255,  80),
}

# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    x1: int; y1: int; x2: int; y2: int
    label: str
    confidence: float
    class_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox":       [self.x1, self.y1, self.x2, self.y2],
            "label":      self.label,
            "confidence": round(self.confidence, 3),
            "class_id":   self.class_id,
            "area":       self.area,
            "timestamp":  datetime.now().isoformat(),
        }

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class FrameStats:
    fps_camera:     float = 0.0
    fps_inference:  float = 0.0
    inference_ms:   float = 0.0
    detections:     List[DetectionResult] = field(default_factory=list)
    monkey_count:   int   = 0
    human_count:    int   = 0
    sequence:       int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps_camera":    round(self.fps_camera, 1),
            "fps_inference": round(self.fps_inference, 1),
            "inference_ms":  round(self.inference_ms, 1),
            "objects":       len(self.detections),
            "monkey_count":  self.monkey_count,
            "human_count":   self.human_count,
            "sequence":      self.sequence,
            "detections":    [d.to_dict() for d in self.detections[:10]],
        }


# ══════════════════════════════════════════════════════════════════════
# SHARED MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════════

class SharedMemoryHelper:

    @staticmethod
    def write_jpeg(shm, lock: threading.Lock, data: bytes) -> bool:
        n = len(data)
        if n > SHM_FRAME_SIZE - 4:
            return False
        with lock:
            struct.pack_into("<I", shm.buf, 0, n)
            shm.buf[4:4 + n] = data
        return True

    @staticmethod
    def read_jpeg(shm, lock: threading.Lock) -> Optional[bytes]:
        with lock:
            try:
                n = struct.unpack_from("<I", shm.buf, 0)[0]
                if n == 0 or n > SHM_FRAME_SIZE - 4:
                    return None
                return bytes(shm.buf[4:4 + n])
            except Exception:
                return None

    @staticmethod
    def write_stats(shm, stats: FrameStats) -> None:
        try:
            packed = struct.pack(
                "<fffBBB",
                stats.fps_camera, stats.fps_inference, stats.inference_ms,
                int(stats.monkey_count > 0),
                int(stats.human_count  > 0),
                stats.sequence % 256,
            )
            shm.buf[:len(packed)] = packed
        except Exception:
            pass

    @staticmethod
    def read_stats(shm) -> Optional[Dict[str, Any]]:
        try:
            vals = struct.unpack_from("<fffBBB", shm.buf, 0)
            return {
                "fps_camera":    round(vals[0], 1),
                "fps_inference": round(vals[1], 1),
                "inference_ms":  round(vals[2], 1),
                "monkey": bool(vals[3]),
                "human":  bool(vals[4]),
                "seq":    vals[5],
            }
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════
# YOLO DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════

class MonkeyDetector:
    """YOLOv8 CPU-optimised detector with correct coordinate scaling."""

    def __init__(
        self,
        model_path: str   = MODEL_PATH_DEFAULT,
        img_size:   int   = YOLO_IMG_SIZE,
        conf_thresh: float= YOLO_CONF_THRESHOLD,
        device:     str   = YOLO_DEVICE,
        use_fp16:   bool  = YOLO_USE_FP16,
    ):
        self.model_path  = model_path
        self.img_size    = img_size
        self.conf_thresh = conf_thresh
        self.device      = device
        self.use_fp16    = use_fp16 and (device == "cuda")

        self.model       = None
        self.class_names : Dict[int, str] = {}
        self._fps_buffer = deque(maxlen=30)
        self._load_lock  = threading.Lock()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed: pip install ultralytics")
            raise

        with self._load_lock:
            logger.info(f"Loading YOLO: {self.model_path}")
            t = time.perf_counter()
            self.model       = YOLO(self.model_path)
            self.model.fuse()
            self.class_names = self.model.model.names
            logger.info(
                f"Model ready in {time.perf_counter()-t:.2f}s | "
                f"classes: {list(self.class_names.values())}"
            )

    def detect(
        self, frame_bgr: np.ndarray
    ) -> Tuple[List[DetectionResult], float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()
        oh, ow = frame_bgr.shape[:2]
        input_frame = cv2.resize(frame_bgr, (self.img_size, self.img_size))

        results = self.model(
            input_frame,
            imgsz   = self.img_size,
            conf    = self.conf_thresh,
            device  = self.device,
            verbose = False,
            half    = self.use_fp16,
        )

        sx, sy = ow / self.img_size, oh / self.img_size
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                x1 = max(0, min(int(x1 * sx), ow - 1))
                y1 = max(0, min(int(y1 * sy), oh - 1))
                x2 = max(0, min(int(x2 * sx), ow - 1))
                y2 = max(0, min(int(y2 * sy), oh - 1))
                cls_id = int(box.cls[0].cpu().numpy())
                conf   = float(box.conf[0].cpu().numpy())
                label  = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append(DetectionResult(x1, y1, x2, y2, label, conf, cls_id))

        inf_ms = (time.perf_counter() - t0) * 1000
        self._fps_buffer.append(inf_ms)
        return detections, inf_ms

    def get_inference_fps(self) -> float:
        if not self._fps_buffer:
            return 0.0
        avg = sum(self._fps_buffer) / len(self._fps_buffer)
        return round(1000.0 / avg, 1) if avg > 0 else 0.0

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None


# ══════════════════════════════════════════════════════════════════════
# COLOUR CORRECTION PIPELINE
# ══════════════════════════════════════════════════════════════════════

class ColourCorrectionPipeline:
    """Histogram-based adaptive colour correction for NoIR cameras."""

    def __init__(
        self,
        ir_attenuation: float  = IR_ATTENUATION,
        blue_correction: float = BLUE_CORRECTION,
        smoothing: float       = HISTOGRAM_SMOOTHING,
    ):
        self.ir_attenuation  = ir_attenuation
        self.blue_correction = blue_correction
        self.smoothing       = smoothing
        self._scale          = {"r": 1.0, "g": 1.0, "b": 1.0}
        self._lock           = threading.Lock()

    def apply(self, frame_rgb: np.ndarray) -> np.ndarray:
        f = frame_rgb.astype(np.float32)
        r, g, b = f[:,:,0], f[:,:,1], f[:,:,2]

        if self.ir_attenuation < 1.0:
            r *= self.ir_attenuation

        avg = (np.mean(r) + np.mean(g) + np.mean(b)) / 3.0
        sr  = avg / (np.mean(r) + 1e-5)
        sg  = avg / (np.mean(g) + 1e-5)
        sb  = (avg / (np.mean(b) + 1e-5)) * self.blue_correction

        with self._lock:
            for key, val in [("r", sr), ("g", sg), ("b", sb)]:
                self._scale[key] = (
                    self.smoothing * val +
                    (1 - self.smoothing) * self._scale[key]
                )
            r *= self._scale["r"]
            g *= self._scale["g"]
            b *= self._scale["b"]

        f[:,:,0], f[:,:,1], f[:,:,2] = r, g, b
        return cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def update_params(
        self,
        ir_attenuation:  Optional[float] = None,
        blue_correction: Optional[float] = None,
        smoothing:       Optional[float] = None,
    ) -> None:
        if ir_attenuation  is not None: self.ir_attenuation  = max(0.0, min(1.0, ir_attenuation))
        if blue_correction is not None: self.blue_correction = max(0.1, min(3.0, blue_correction))
        if smoothing       is not None: self.smoothing       = max(0.0, min(1.0, smoothing))


# ══════════════════════════════════════════════════════════════════════
# ALERT MANAGER  (video side — rate-limits calls to trigger_alert)
# ══════════════════════════════════════════════════════════════════════

class AlertManager:
    """Prevents alert flooding; calls trigger_alert() via alert_client."""

    def __init__(self, cooldown: float = ALERT_COOLDOWN_SECONDS):
        self.cooldown       = cooldown
        self._last_alert    = 0.0
        self._lock          = threading.Lock()

    def check_and_fire(
        self,
        monkey_detections: List[DetectionResult],
        human_present: bool,
    ) -> None:
        if not monkey_detections:
            return

        now = time.time()
        with self._lock:
            if now - self._last_alert < self.cooldown:
                return
            self._last_alert = now

        max_conf = max(d.confidence for d in monkey_detections) * 100
        ts       = datetime.now().strftime("%H:%M:%S")
        tag      = "  ⚠ HUMAN IN FRAME" if human_present else ""
        logger.warning(
            f"\n{'═'*50}\n"
            f"🐒 MONKEY DETECTED [{ts}]{tag}\n"
            f"   Count: {len(monkey_detections)} | Conf: {max_conf:.1f}%\n"
            f"{'═'*50}"
        )
        # Send to alert_service
        trigger_alert(source="video", confidence=max_conf, human=human_present)


# ══════════════════════════════════════════════════════════════════════
# FRAME PROCESSOR
# ══════════════════════════════════════════════════════════════════════

class FrameProcessor:
    """Colour correction + detection + annotation in one pass."""

    def __init__(
        self,
        detector:         MonkeyDetector,
        colour_pipeline:  ColourCorrectionPipeline,
        alert_cooldown:   float = ALERT_COOLDOWN_SECONDS,
    ):
        self.detector        = detector
        self.colour_pipeline = colour_pipeline
        self.alert_manager   = AlertManager(alert_cooldown)

        self._frame_lock     = threading.Lock()
        self._latest_frame   = None
        self._latest_stats   = FrameStats()
        self._sequence       = 0
        self._cam_fps_buf    = deque(maxlen=30)
        self._last_cam_time  = time.perf_counter()

    def process_frame(
        self, frame_rgb: np.ndarray
    ) -> Tuple[np.ndarray, FrameStats]:
        now           = time.perf_counter()
        delta         = now - self._last_cam_time
        self._cam_fps_buf.append(delta)
        self._last_cam_time = now
        fps_cam = 1.0 / (sum(self._cam_fps_buf) / len(self._cam_fps_buf)) if self._cam_fps_buf else 0.0

        frame_bgr          = self.colour_pipeline.apply(frame_rgb)
        detections, inf_ms = self.detector.detect(frame_bgr)

        monkey_count = sum(1 for d in detections if d.label == "Monkey")
        human_count  = sum(1 for d in detections if d.label == "Human")
        self._sequence += 1

        stats = FrameStats(
            fps_camera    = fps_cam,
            fps_inference = self.detector.get_inference_fps(),
            inference_ms  = inf_ms,
            detections    = detections,
            monkey_count  = monkey_count,
            human_count   = human_count,
            sequence      = self._sequence,
        )

        annotated = self._draw_annotations(frame_bgr, detections, human_count > 0)
        annotated = self._add_hud(annotated, stats)

        # Fire alert (rate-limited) via Unix socket → alert_service
        self.alert_manager.check_and_fire(
            [d for d in detections if d.label == "Monkey"],
            human_present=(human_count > 0),
        )

        with self._frame_lock:
            self._latest_frame = annotated.copy()
            self._latest_stats = stats

        return annotated, stats

    def _draw_annotations(
        self,
        frame_bgr: np.ndarray,
        detections: List[DetectionResult],
        human_present: bool,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        fh, fw = out.shape[:2]

        for det in detections:
            if det.x1 >= det.x2 or det.y1 >= det.y2:
                continue
            x1 = max(0, min(det.x1, fw - 1))
            y1 = max(0, min(det.y1, fh - 1))
            x2 = max(0, min(det.x2, fw - 1))
            y2 = max(0, min(det.y2, fh - 1))

            if det.label == "Human":
                color = COLORS["Human"]
            elif det.label == "Monkey":
                color = COLORS["Monkey_Warn"] if human_present else COLORS["Monkey"]
            else:
                color = COLORS["Other"]

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            label_text = f"{det.label} {det.confidence*100:.1f}%"
            fs, ft     = 0.6, 2
            (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)

            bg_y1 = y1 - th - bl - 4
            bg_y2 = y1
            ly    = y1 - bl

            if bg_y1 < 0:
                bg_y1 = y2 + 2
                bg_y2 = y2 + th + bl + 4
                ly    = y2 + th + bl

            ov = out.copy()
            cv2.rectangle(ov, (x1, bg_y1), (x1 + tw + 4, bg_y2), color, cv2.FILLED)
            cv2.addWeighted(ov, 0.7, out, 0.3, 0, out)
            cv2.putText(out, label_text, (x1 + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)

            # Confidence bar under monkey boxes
            if det.label == "Monkey":
                bw = int((x2 - x1) * det.confidence)
                by = y2 + 3
                if by + 5 < fh:
                    cv2.rectangle(out, (x1, by), (x2, by + 5), (50, 50, 50), cv2.FILLED)
                    if bw > 0:
                        cv2.rectangle(out, (x1, by), (x1 + bw, by + 5), color, cv2.FILLED)

        return out

    def _add_hud(self, frame_bgr: np.ndarray, stats: FrameStats) -> np.ndarray:
        out  = frame_bgr.copy()
        text = (f"FPS:{stats.fps_camera:.0f} | INF:{stats.inference_ms:.0f}ms | "
                f"🐒{stats.monkey_count} 👤{stats.human_count}")
        ov   = out.copy()
        cv2.rectangle(ov, (8, 8), (360, 42), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(ov, 0.65, out, 0.35, 0, out)
        cv2.putText(out, text, (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["HUD_Text"], 2, cv2.LINE_AA)
        return out

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_stats(self) -> FrameStats:
        with self._frame_lock:
            return self._latest_stats


# ══════════════════════════════════════════════════════════════════════
# FLASK WEB APPLICATION
# ══════════════════════════════════════════════════════════════════════

def create_flask_app(
    processor:        FrameProcessor,
    shm_jpeg,
    shm_stats,
    jpeg_lock:        threading.Lock,
    colour_pipeline:  ColourCorrectionPipeline,
) -> Flask:
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/video_feed')
    def video_feed():
        def generate():
            last_seq = -1
            while True:
                stats = SharedMemoryHelper.read_stats(shm_stats)
                if stats and stats["seq"] != last_seq:
                    data = SharedMemoryHelper.read_jpeg(shm_jpeg, jpeg_lock)
                    if data:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
                        last_seq = stats["seq"]
                else:
                    time.sleep(0.01)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/get_settings')
    def get_settings():
        return jsonify({
            "exposure":       current_camera_settings["exposure"],
            "gain":           current_camera_settings["gain"],
            "red_gain":       current_camera_settings["red_gain"],
            "blue_gain":      current_camera_settings["blue_gain"],
            "ir_attenuation": colour_pipeline.ir_attenuation,
            "blue_correction":colour_pipeline.blue_correction,
            "smoothing":      colour_pipeline.smoothing,
        })

    @app.route('/set_exposure', methods=['POST'])
    def set_exposure():
        try:
            exp = int(request.get_json().get('exposure', 20000))
            if not (1000 <= exp <= 200000):
                return jsonify({"status": "❌ Out of range (1000-200000 μs)"}), 400
            picam.set_controls({"ExposureTime": exp})
            current_camera_settings["exposure"] = exp
            return jsonify({"status": f"✅ Exposure → {exp} μs"})
        except Exception as e:
            return jsonify({"status": f"❌ {e}"}), 400

    @app.route('/set_gain', methods=['POST'])
    def set_gain():
        try:
            gain = float(request.get_json().get('gain', 1.0))
            if not (0.1 <= gain <= 8.0):
                return jsonify({"status": "❌ Out of range (0.1-8.0)"}), 400
            picam.set_controls({"AnalogueGain": gain})
            current_camera_settings["gain"] = gain
            return jsonify({"status": f"✅ Gain → {gain}"})
        except Exception as e:
            return jsonify({"status": f"❌ {e}"}), 400

    @app.route('/set_colour_gains', methods=['POST'])
    def set_colour_gains():
        try:
            data = request.get_json()
            red  = float(data.get('red_gain',  1.0))
            blue = float(data.get('blue_gain', 1.0))
            if not (0.1 <= red <= 8.0 and 0.1 <= blue <= 8.0):
                return jsonify({"status": "❌ Out of range (0.1-8.0)"}), 400
            picam.set_controls({"ColourGains": (red, blue)})
            current_camera_settings.update(red_gain=red, blue_gain=blue)
            save_calibration(red, blue)
            return jsonify({"status": "✅ Colour gains updated"})
        except Exception as e:
            return jsonify({"status": f"❌ {e}"}), 400

    @app.route('/set_colour_params', methods=['POST'])
    def set_colour_params():
        try:
            data = request.get_json()
            colour_pipeline.update_params(
                ir_attenuation  = data.get('ir_attenuation'),
                blue_correction = data.get('blue_correction'),
                smoothing       = data.get('smoothing'),
            )
            return jsonify({"status": "✅ Pipeline updated"})
        except Exception as e:
            return jsonify({"status": f"❌ {e}"}), 400

    @app.route('/api/stats')
    def api_stats():
        return jsonify(processor.get_latest_stats().to_dict())

    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

    return app


# ══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🐒 Monkey Detector — NoIR V2 + YOLO</title>
<style>
:root {
    --bg:#0a0e14;--panel:#111827;--border:#1f2937;
    --accent:#10b981;--accent2:#3b82f6;--danger:#ef4444;
    --text:#f1f5f9;--muted:#94a3b8;--mono:'SF Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--mono);background:var(--bg);color:var(--text);padding:16px;min-height:100vh}
.container{max-width:1400px;margin:0 auto;display:grid;grid-template-columns:1fr 320px;gap:16px}
.header{grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;
    padding:12px 16px;background:var(--panel);border:1px solid var(--border);border-radius:8px;margin-bottom:16px}
.logo{font-size:1.2rem;font-weight:700;color:var(--accent)}
.badge{padding:4px 12px;background:var(--accent);color:#000;border-radius:20px;font-size:.75rem;font-weight:600}
.video-panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
.video-container{position:relative;background:#000;border-radius:6px;overflow:hidden;aspect-ratio:16/9}
#videoFeed{width:100%;height:100%;object-fit:contain;display:block}
.stats-bar{display:flex;gap:12px;padding:8px 0;flex-wrap:wrap}
.stat{background:var(--border);padding:6px 12px;border-radius:4px;font-size:.85rem}
.stat .label{color:var(--muted)} .stat .value{color:var(--accent);font-weight:600}
.control-panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;
    padding:16px;display:flex;flex-direction:column;gap:16px;height:fit-content}
.section{border-bottom:1px solid var(--border);padding-bottom:12px}
.section:last-child{border-bottom:none}
.section-title{font-size:.9rem;color:var(--accent2);font-weight:600;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em}
.control-row{display:flex;align-items:center;gap:8px;margin:8px 0}
.control-row label{min-width:90px;color:var(--muted);font-size:.8rem}
.control-row input{flex:1;padding:6px 10px;background:var(--border);border:1px solid var(--border);
    border-radius:4px;color:var(--text);font-family:var(--mono);font-size:.85rem}
.control-row input[type="range"]{padding:0;height:4px}
.btn{padding:8px 16px;background:var(--accent);color:#000;border:none;border-radius:4px;
    font-weight:600;cursor:pointer;font-family:var(--mono);font-size:.85rem;transition:opacity .2s}
.btn:hover{opacity:.9} .btn.secondary{background:var(--border);color:var(--text)}
.btn-row{display:flex;gap:8px;margin-top:8px}
.status{padding:8px 12px;background:var(--border);border-radius:4px;font-size:.8rem;text-align:center;min-height:36px}
.status.success{background:rgba(16,185,129,.2);color:var(--accent)}
.status.error{background:rgba(239,68,68,.2);color:var(--danger)}
.alert-indicator{position:absolute;top:12px;right:12px;padding:6px 12px;background:var(--danger);
    color:#fff;border-radius:20px;font-size:.75rem;font-weight:600;opacity:0;transition:opacity .3s;pointer-events:none}
.alert-indicator.active{opacity:1;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
@media(max-width:1024px){.container{grid-template-columns:1fr}.control-panel{order:-1}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="logo">🐒 Monkey Detector <span style="color:var(--muted)">//</span> NoIR V2 + YOLO</div>
    <div class="badge">● LIVE</div>
  </div>
  <div class="video-panel">
    <div class="video-container">
      <img id="videoFeed" src="/video_feed" alt="Live stream">
      <div class="alert-indicator" id="alertIndicator">🐒 MONKEY ALERT</div>
    </div>
    <div class="stats-bar">
      <span class="stat"><span class="label">CAM:</span> <span class="value" id="fpsCam">--</span> fps</span>
      <span class="stat"><span class="label">INF:</span> <span class="value" id="fpsInf">--</span> fps</span>
      <span class="stat"><span class="label">LAT:</span> <span class="value" id="infMs">--</span> ms</span>
      <span class="stat"><span class="label">🐒:</span> <span class="value" id="monkeyCount">0</span></span>
      <span class="stat"><span class="label">👤:</span> <span class="value" id="humanCount">0</span></span>
    </div>
  </div>
  <div class="control-panel">
    <div class="section">
      <div class="section-title">📷 Camera</div>
      <div class="control-row"><label>Exposure (μs):</label><input type="number" id="expInput" min="1000" max="200000" step="1000" value="20000"></div>
      <div class="control-row"><label>Analogue Gain:</label><input type="number" id="gainInput" min="0.1" max="8.0" step="0.1" value="1.0"></div>
      <div class="btn-row"><button class="btn" onclick="applyCameraSettings()">Apply</button><button class="btn secondary" onclick="loadSettings()">Refresh</button></div>
    </div>
    <div class="section">
      <div class="section-title">🎨 Colour Gains</div>
      <div class="control-row"><label>Red Gain:</label><input type="number" id="redGainInput" min="0.1" max="8.0" step="0.1" value="1.0"></div>
      <div class="control-row"><label>Blue Gain:</label><input type="number" id="blueGainInput" min="0.1" max="8.0" step="0.1" value="1.0"></div>
      <div class="btn-row"><button class="btn" onclick="applyColourGains()">Apply</button></div>
    </div>
    <div class="section">
      <div class="section-title">⚙️ Colour Pipeline</div>
      <div class="control-row"><label>IR Atten:</label><input type="range" id="irAttInput" min="0.1" max="1.0" step="0.05" value="0.5"><span id="irAttVal">0.50</span></div>
      <div class="control-row"><label>Blue Corr:</label><input type="range" id="blueCorrInput" min="0.5" max="2.0" step="0.05" value="1.15"><span id="blueCorrVal">1.15</span></div>
      <div class="btn-row"><button class="btn" onclick="applyColourParams()">Apply</button><button class="btn secondary" onclick="resetColourParams()">Reset</button></div>
    </div>
    <div id="status" class="status">✅ System ready</div>
  </div>
</div>
<script>
function syncRange(id,disp){document.getElementById(id).addEventListener('input',()=>document.getElementById(disp).textContent=parseFloat(document.getElementById(id).value).toFixed(2))}
syncRange('irAttInput','irAttVal');syncRange('blueCorrInput','blueCorrVal');
function showStatus(msg,type='info'){const el=document.getElementById('status');el.textContent=msg;el.className='status '+(type==='error'?'error':type==='success'?'success':'');setTimeout(()=>{if(el.textContent===msg){el.textContent='✅ System ready';el.className='status'}},3000)}
async function loadSettings(){try{const d=await(await fetch('/get_settings')).json();document.getElementById('expInput').value=d.exposure;document.getElementById('gainInput').value=d.gain;document.getElementById('redGainInput').value=d.red_gain;document.getElementById('blueGainInput').value=d.blue_gain;document.getElementById('irAttInput').value=d.ir_attenuation;document.getElementById('blueCorrInput').value=d.blue_correction;document.getElementById('irAttVal').textContent=d.ir_attenuation.toFixed(2);document.getElementById('blueCorrVal').textContent=d.blue_correction.toFixed(2);showStatus('Settings loaded','success')}catch(e){showStatus('Failed to load','error')}}
async function applyCameraSettings(){const exp=parseInt(document.getElementById('expInput').value),gain=parseFloat(document.getElementById('gainInput').value);if(exp<1000||exp>200000){showStatus('Exposure 1000-200000 μs','error');return}if(gain<0.1||gain>8){showStatus('Gain 0.1-8.0','error');return}try{await fetch('/set_exposure',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({exposure:exp})});await fetch('/set_gain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gain})});showStatus('Camera applied','success')}catch(e){showStatus('Failed','error')}}
async function applyColourGains(){const r=parseFloat(document.getElementById('redGainInput').value),b=parseFloat(document.getElementById('blueGainInput').value);if(r<0.1||r>8||b<0.1||b>8){showStatus('Gains 0.1-8.0','error');return}try{const res=await fetch('/set_colour_gains',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({red_gain:r,blue_gain:b})});showStatus((await res.json()).status,res.ok?'success':'error')}catch(e){showStatus('Failed','error')}}
async function applyColourParams(){const ir=parseFloat(document.getElementById('irAttInput').value),bl=parseFloat(document.getElementById('blueCorrInput').value);try{const res=await fetch('/set_colour_params',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ir_attenuation:ir,blue_correction:bl})});showStatus((await res.json()).status,res.ok?'success':'error')}catch(e){showStatus('Failed','error')}}
function resetColourParams(){document.getElementById('irAttInput').value=0.5;document.getElementById('blueCorrInput').value=1.15;document.getElementById('irAttVal').textContent='0.50';document.getElementById('blueCorrVal').textContent='1.15';applyColourParams()}
async function pollStats(){try{const d=await(await fetch('/api/stats')).json();document.getElementById('fpsCam').textContent=d.fps_camera;document.getElementById('fpsInf').textContent=d.fps_inference;document.getElementById('infMs').textContent=d.inference_ms;document.getElementById('monkeyCount').textContent=d.monkey_count;document.getElementById('humanCount').textContent=d.human_count;document.getElementById('alertIndicator').classList.toggle('active',d.monkey_count>0)}catch(e){}}
loadSettings();setInterval(pollStats,500);
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
# CAMERA & CALIBRATION
# ══════════════════════════════════════════════════════════════════════

picam = None
current_camera_settings = {"exposure": 20000, "gain": 1.0, "red_gain": 1.0, "blue_gain": 1.0}


def load_calibration(filepath: str = CALIB_FILE) -> Dict[str, float]:
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                data = json.load(f)
                return {"red_gain": data.get("red_gain", 1.0), "blue_gain": data.get("blue_gain", 1.0)}
        except Exception:
            pass
    return {"red_gain": 1.0, "blue_gain": 1.0}


def save_calibration(red: float, blue: float, filepath: str = CALIB_FILE) -> None:
    try:
        with open(filepath, "w") as f:
            json.dump({"red_gain": red, "blue_gain": blue}, f, indent=2)
    except Exception as e:
        logger.error(f"Calibration save failed: {e}")


def initialize_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cal = load_calibration()
    current_camera_settings.update(red_gain=cal["red_gain"], blue_gain=cal["blue_gain"])

    cfg = cam.create_video_configuration(
        main    = {"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
        buffer_count = 4,
        controls= {
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": current_camera_settings["exposure"],
            "AnalogueGain": current_camera_settings["gain"],
            "Saturation": 1.1, "Contrast": 1.05, "Sharpness": 0.3,
        },
    )
    cam.configure(cfg)
    cam.set_controls({"ColourGains": (cal["red_gain"], cal["blue_gain"]), "AwbEnable": False})
    cam.start()
    time.sleep(1.0)
    logger.info(f"Camera ready: {CAM_WIDTH}x{CAM_HEIGHT}")
    return cam


# ══════════════════════════════════════════════════════════════════════
# CAPTURE THREAD
# ══════════════════════════════════════════════════════════════════════

def capture_thread_func(camera, processor, shm_jpeg, shm_stats, jpeg_lock, stop_event):
    logger.info("Capture thread started.")
    while not stop_event.is_set():
        try:
            frame_rgb = camera.capture_array("main")
            annotated, stats = processor.process_frame(frame_rgb)
            ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                SharedMemoryHelper.write_jpeg(shm_jpeg, jpeg_lock, buf.tobytes())
                SharedMemoryHelper.write_stats(shm_stats, stats)
        except Exception as e:
            logger.error(f"Capture error: {e}", exc_info=True)
            time.sleep(0.1)
    logger.info("Capture thread stopped.")


# ══════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NoIR V2 + YOLO Flask Server")
    p.add_argument("--model",   "-m", default=MODEL_PATH_DEFAULT)
    p.add_argument("--imgsize", "-s", type=int, default=YOLO_IMG_SIZE,
                   choices=[128, 160, 192, 224, 256])
    p.add_argument("--conf",    "-c", type=float, default=YOLO_CONF_THRESHOLD)
    p.add_argument("--port",    "-p", type=int, default=5000)
    p.add_argument("--host",          default="0.0.0.0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger.info(f"Starting | model={args.model} imgsize={args.imgsize} port={args.port}")

    shm_jpeg = shm_stats = None
    jpeg_lock  = threading.Lock()
    stop_event = threading.Event()

    try:
        from multiprocessing import shared_memory
        shm_jpeg  = shared_memory.SharedMemory(create=True, size=SHM_FRAME_SIZE, name=SHM_JPEG_NAME)
        shm_stats = shared_memory.SharedMemory(create=True, size=SHM_STATS_SIZE, name=SHM_STATS_NAME)

        global picam
        picam = initialize_camera()

        detector  = MonkeyDetector(model_path=args.model, img_size=args.imgsize, conf_thresh=args.conf)
        colour    = ColourCorrectionPipeline()
        processor = FrameProcessor(detector=detector, colour_pipeline=colour)
        app       = create_flask_app(processor, shm_jpeg, shm_stats, jpeg_lock, colour)

        t = threading.Thread(
            target=capture_thread_func,
            args=(picam, processor, shm_jpeg, shm_stats, jpeg_lock, stop_event),
            name="capture", daemon=True,
        )
        t.start()

        print(f"\n{'═'*60}")
        print(f"🐒 Monkey Detector — NoIR V2 + YOLOv8")
        print(f"{'═'*60}")
        print(f"🌐 Web UI  : http://localhost:{args.port}")
        print(f"📡 Stream  : http://<Pi-IP>:{args.port}/video_feed")
        print(f"📊 Stats   : http://<Pi-IP>:{args.port}/api/stats")
        print(f"🔔 Alerts  → alert_service via /tmp/monkey_alert.sock")
        print(f"{'═'*60}\n")

        app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        return 1
    finally:
        stop_event.set()
        if picam:
            picam.stop(); picam.close()
        for shm in (shm_jpeg, shm_stats):
            if shm:
                try: shm.close(); shm.unlink()
                except Exception: pass
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    sys.exit(main())
