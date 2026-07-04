# """
# ╔══════════════════════════════════════════════════════════╗
# ║     MONKEY WARNING SYSTEM — audio_handler.py            ║
# ║  INMP441 I2S Reader + ML Inference Pipeline             ║
# ╚══════════════════════════════════════════════════════════╝

# Handles:
#   • INMP441Reader   — I2S microphone capture (stereo→mono, normalisation)
#   • AudioCapture    — background thread that fills _audio_q
#   • InferencePipeline — YAMNet + custom classifier, fills _infer_q
# """

# import os
# import sys
# import queue
# import logging
# import warnings
# from contextlib import contextmanager
# from pathlib import Path

# import numpy as np

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# import pyaudio
# import librosa
# import tensorflow as tf
# import tensorflow_hub as hub

# log = logging.getLogger("MonkeyWarning")


# # ══════════════════════════════════════════════════════════════════════
# # STDERR SUPPRESSOR  (silences ALSA/JACK spam on open)
# # ══════════════════════════════════════════════════════════════════════

# @contextmanager
# def _suppress_stderr():
#     devnull_fd = os.open("/dev/null", os.O_WRONLY)
#     saved_fd   = os.dup(2)
#     try:
#         os.dup2(devnull_fd, 2)
#         yield
#     finally:
#         os.dup2(saved_fd, 2)
#         os.close(saved_fd)
#         os.close(devnull_fd)


# # ══════════════════════════════════════════════════════════════════════
# # AUDIO CONFIGURATION  (subset used by this module)
# # ══════════════════════════════════════════════════════════════════════

# class AudioConfig:
#     SAMPLE_RATE        = 48000
#     YAMNET_RATE        = 16000
#     CHANNELS           = 2          # stereo driver; left ch extracted
#     CHUNK              = 1024
#     RECORD_SECONDS     = 1.5
#     PREFERRED_FORMAT   = pyaudio.paInt32
#     FALLBACK_FORMAT    = pyaudio.paInt16
#     INFERENCE_GAIN     = 1.15
#     RMS_SILENCE_GATE   = 0.0001
#     AUDIO_DEVICE_INDEX = 1          # hw:1,0

#     DETECTION_THRESHOLD = 0.65
#     REQUIRED_STREAK     = 3

#     MODEL_PATH       = "/home/monkeyaudio/monkey/video/best_model.h5"
#     CLASS_NAMES_PATH = "/home/monkeyaudio/monkey/video/class_names.txt"
#     YAMNET_PATH      = "yamnet"


# # ══════════════════════════════════════════════════════════════════════
# # INMP441 I2S AUDIO READER
# # ══════════════════════════════════════════════════════════════════════

# class INMP441Reader:
#     """
#     Reads from the INMP441 via the GoogleVoiceHAT I2S driver (hw:1,0).

#     The driver exposes a 2-channel (stereo) device even though the
#     INMP441 is physically mono.  Real audio lands on the LEFT channel.
#     read() strips the interleaved stereo by keeping every other sample
#     (samples[::2]) so the rest of the pipeline gets a clean mono
#     float32 array normalised to [-1.0, 1.0].
#     """

#     def __init__(self, pa: pyaudio.PyAudio):
#         self._pa     = pa
#         self._stream = None
#         self._dtype  = None
#         self._scale  = None
#         self._shift  = None
#         self._open_stream()

#     def _open_stream(self) -> None:
#         """
#         Try paInt32 first (true 24-bit depth from the INMP441),
#         fall back to paInt16 if the driver refuses 32-bit.
#         """
#         for fmt, dtype, scale, shift in [
#             (AudioConfig.PREFERRED_FORMAT, np.int32, float(1 << 23), 8),
#             (AudioConfig.FALLBACK_FORMAT,  np.int16, float(1 << 15), 0),
#         ]:
#             try:
#                 with _suppress_stderr():
#                     stream = self._pa.open(
#                         format             = fmt,
#                         channels           = AudioConfig.CHANNELS,
#                         rate               = AudioConfig.SAMPLE_RATE,
#                         input              = True,
#                         input_device_index = AudioConfig.AUDIO_DEVICE_INDEX,
#                         frames_per_buffer  = AudioConfig.CHUNK,
#                     )
#                 self._stream = stream
#                 self._dtype  = dtype
#                 self._scale  = scale
#                 self._shift  = shift
#                 label = ("paInt32 (24-bit)" if fmt == pyaudio.paInt32
#                          else "paInt16 fallback")
#                 log.info(
#                     f"[INMP441] Device {AudioConfig.AUDIO_DEVICE_INDEX} | "
#                     f"{AudioConfig.SAMPLE_RATE} Hz | {AudioConfig.CHANNELS}ch | {label}"
#                 )
#                 return
#             except Exception as exc:
#                 log.warning(f"[INMP441] Format failed: {exc}")

#         raise RuntimeError("[INMP441] Could not open audio stream.")

#     def read(self, n_chunks: int) -> np.ndarray:
#         """
#         Read n_chunks × CHUNK frames and return a mono float32 array
#         covering RECORD_SECONDS of audio.
#         """
#         raw = b"".join(
#             self._stream.read(AudioConfig.CHUNK, exception_on_overflow=False)
#             for _ in range(n_chunks)
#         )
#         samples = np.frombuffer(raw, dtype=self._dtype)

#         # Stereo interleaved → left channel only
#         if AudioConfig.CHANNELS == 2:
#             samples = samples[::2]

#         # paInt32 from this driver is left-justified; shift down to 24-bit
#         if self._shift:
#             samples = samples >> self._shift

#         return samples.astype(np.float32) / self._scale

#     def close(self) -> None:
#         try:
#             if self._stream and self._stream.is_active():
#                 self._stream.stop_stream()
#             if self._stream:
#                 self._stream.close()
#         except Exception:
#             pass


# # ══════════════════════════════════════════════════════════════════════
# # INFERENCE PIPELINE
# # ══════════════════════════════════════════════════════════════════════

# class InferencePipeline:
#     """
#     Loads YAMNet and the custom classifier.
#     Provides:
#       • audio_thread()     — reads mic, pushes frames to audio_q
#       • inference_thread() — pops frames, runs models, pushes results to infer_q
#     Both are designed to run as daemon threads managed by MonkeyWarningSystem.
#     """

#     def __init__(self):
#         self._running  = False
#         self.audio_q: queue.Queue = queue.Queue(maxsize=10)
#         self.infer_q:  queue.Queue = queue.Queue(maxsize=5)
#         self.class_names: list[str] = []

#         self._validate_paths()
#         self._load_models()

#         with _suppress_stderr():
#             self._pa  = pyaudio.PyAudio()
#         self._mic = INMP441Reader(self._pa)

#     # ── Startup validation ─────────────────────────────────────────────

#     @staticmethod
#     def _validate_paths() -> None:
#         checks = {
#             "Model file":       AudioConfig.MODEL_PATH,
#             "Class names file": AudioConfig.CLASS_NAMES_PATH,
#         }
#         failed = False
#         for label, path in checks.items():
#             if not Path(path).exists():
#                 log.error(f"[Startup] {label} not found: {path}")
#                 failed = True
#         if failed:
#             log.error("[Startup] Missing audio model paths — aborting.")
#             sys.exit(1)
#         log.info("[Startup] Audio model paths validated OK.")

#     # ── Model loading ──────────────────────────────────────────────────

#     def _load_models(self) -> None:
#         log.info("[Audio] Loading YAMNet …")
#         self._yamnet = hub.load(AudioConfig.YAMNET_PATH)
#         log.info("[Audio] YAMNet loaded.")

#         log.info("[Audio] Loading custom classifier …")
#         self._clf = tf.keras.models.load_model(AudioConfig.MODEL_PATH)

#         with open(AudioConfig.CLASS_NAMES_PATH) as fh:
#             self.class_names = [l.strip() for l in fh]
#         log.info(f"[Audio] Classes: {self.class_names}")

#     # ── Thread: Audio Capture ──────────────────────────────────────────

#     def audio_thread(self) -> None:
#         """
#         Continuously reads mic frames and enqueues them for inference.
#         Frames below RMS_SILENCE_GATE are discarded to avoid wasting
#         inference cycles on silence.
#         """
#         log.info("[AudioThread] Started.")
#         n_chunks = int(
#             AudioConfig.SAMPLE_RATE / AudioConfig.CHUNK * AudioConfig.RECORD_SECONDS
#         )

#         while self._running:
#             try:
#                 audio = self._mic.read(n_chunks)
#             except OSError as exc:
#                 if self._running:
#                     log.error(f"[AudioThread] Read error: {exc}")
#                 continue

#             rms = float(np.sqrt(np.mean(audio ** 2)))
#             if rms <= AudioConfig.RMS_SILENCE_GATE:
#                 continue

#             peak  = np.max(np.abs(audio)) + 1e-6
#             normd = np.clip(
#                 (audio / peak) * AudioConfig.INFERENCE_GAIN, -1.0, 1.0
#             )

#             try:
#                 self.audio_q.put_nowait(normd)
#             except queue.Full:
#                 pass   # inference thread is backed up; drop frame

#         log.info("[AudioThread] Stopped.")

#     # ── Thread: ML Inference ───────────────────────────────────────────

#     def inference_thread(self) -> None:
#         """
#         Pops audio frames from audio_q, runs YAMNet → embeddings →
#         custom classifier → monkey probability, pushes result to infer_q.
#         """
#         log.info("[InferenceThread] Started.")

#         while self._running:
#             try:
#                 audio_48k = self.audio_q.get(timeout=1.0)
#             except queue.Empty:
#                 continue

#             audio_16k = librosa.resample(
#                 audio_48k,
#                 orig_sr=AudioConfig.SAMPLE_RATE,
#                 target_sr=AudioConfig.YAMNET_RATE,
#             )

#             _, embeddings, _ = self._yamnet(audio_16k)

#             prob = float(
#                 self._clf.predict(
#                     embeddings.numpy()[0].reshape(1, -1), verbose=0
#                 )[0][0]
#             )

#             try:
#                 self.infer_q.put_nowait({"prob": prob})
#             except queue.Full:
#                 pass

#         log.info("[InferenceThread] Stopped.")

#     # ── Lifecycle ──────────────────────────────────────────────────────

#     def start(self) -> None:
#         self._running = True

#     def stop(self) -> None:
#         self._running = False
#         self._mic.close()
#         self._pa.terminate()
#         log.info("[Audio] Pipeline stopped.")

"""
Audio.py — INMP441 I2S Microphone + YAMNet ML Inference
════════════════════════════════════════════════════════════════════════════
Runs inside the venv (needs tensorflow + numpy >= 2.0).
Detects monkey sounds and sends trigger to alert_service via Unix socket.

Run via venv:
    /home/monkeyaudio/monkey/video/venv/bin/python3 Audio.py

systemd service example:
    ExecStart=/home/monkeyaudio/monkey/video/venv/bin/python3 \
              /home/monkeyaudio/monkey/video/Audio.py
"""

import os
import sys
import queue
import logging
import warnings
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pyaudio
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# alert_client lives next to this file — works in the venv with no extra deps
from alert_client import trigger_alert

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.DEBUG,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("MonkeyWarning")


# ══════════════════════════════════════════════════════════════════════
# STDERR SUPPRESSOR  — silences ALSA/JACK noise on stream open
# ══════════════════════════════════════════════════════════════════════

@contextmanager
def _suppress_stderr():
    devnull_fd = os.open("/dev/null", os.O_WRONLY)
    saved_fd   = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

class AudioConfig:
    # ── Microphone ────────────────────────────────────────────────────
    SAMPLE_RATE        = 48000
    YAMNET_RATE        = 16000
    CHANNELS           = 2           # stereo I2S driver; left ch extracted
    CHUNK              = 1024
    RECORD_SECONDS     = 1.5
    PREFERRED_FORMAT   = pyaudio.paInt32
    FALLBACK_FORMAT    = pyaudio.paInt16
    INFERENCE_GAIN     = 1.15
    RMS_SILENCE_GATE   = 0.0001
    AUDIO_DEVICE_INDEX = 1           # hw:1,0  (snd_rpi_googlevoicehat_soundcar)

    # ── ML Detection ──────────────────────────────────────────────────
    DETECTION_THRESHOLD = 0.60     # probability threshold [0-1]
    REQUIRED_STREAK     = 3        # consecutive hits before alarm fires

    # ── Model Paths ───────────────────────────────────────────────────
    MODEL_PATH       = "/home/monkeyaudio/monkey/video/best_model.h5"
    CLASS_NAMES_PATH = "/home/monkeyaudio/monkey/video/class_names.txt"
    YAMNET_PATH      = "yamnet"      # TF Hub handle or local path


# ══════════════════════════════════════════════════════════════════════
# INMP441 I2S MICROPHONE READER
# ══════════════════════════════════════════════════════════════════════

class INMP441Reader:
    """
    Reads from the INMP441 via the GoogleVoiceHAT I2S driver (hw:1,0).

    The driver is stereo even though the INMP441 is mono.
    Real audio lands on the LEFT channel — read() keeps samples[::2]
    and returns a normalised mono float32 array in [-1.0, 1.0].
    """

    def __init__(self, pa: pyaudio.PyAudio):
        self._pa    = pa
        self._stream = None
        self._dtype  = None
        self._scale  = None
        self._shift  = None
        self._open_stream()

    def _open_stream(self) -> None:
        for fmt, dtype, scale, shift in [
            (AudioConfig.PREFERRED_FORMAT, np.int32, float(1 << 23), 8),
            (AudioConfig.FALLBACK_FORMAT,  np.int16, float(1 << 15), 0),
        ]:
            try:
                with _suppress_stderr():
                    stream = self._pa.open(
                        format             = fmt,
                        channels           = AudioConfig.CHANNELS,
                        rate               = AudioConfig.SAMPLE_RATE,
                        input              = True,
                        input_device_index = AudioConfig.AUDIO_DEVICE_INDEX,
                        frames_per_buffer  = AudioConfig.CHUNK,
                    )
                self._stream = stream
                self._dtype  = dtype
                self._scale  = scale
                self._shift  = shift
                label = "paInt32 (24-bit)" if fmt == pyaudio.paInt32 else "paInt16 fallback"
                log.info(
                    f"[INMP441] Device {AudioConfig.AUDIO_DEVICE_INDEX} | "
                    f"{AudioConfig.SAMPLE_RATE} Hz | {AudioConfig.CHANNELS}ch | {label}"
                )
                return
            except Exception as exc:
                log.warning(f"[INMP441] Format failed: {exc}")

        raise RuntimeError("[INMP441] Could not open audio stream.")

    def read(self, n_chunks: int) -> np.ndarray:
        raw = b"".join(
            self._stream.read(AudioConfig.CHUNK, exception_on_overflow=False)
            for _ in range(n_chunks)
        )
        samples = np.frombuffer(raw, dtype=self._dtype)
        if AudioConfig.CHANNELS == 2:
            samples = samples[::2]                  # left channel only
        if self._shift:
            samples = samples >> self._shift        # left-justified 32-bit → 24-bit
        return samples.astype(np.float32) / self._scale

    def close(self) -> None:
        try:
            if self._stream and self._stream.is_active():
                self._stream.stop_stream()
            if self._stream:
                self._stream.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# MONKEY AUDIO DETECTOR
# ══════════════════════════════════════════════════════════════════════

class MonkeyAudioDetector:
    """
    Orchestrates:
      • INMP441 mic → audio frames
      • YAMNet embeddings
      • Custom classifier → monkey probability
      • Streak logic → trigger_alert() via alert_client

    Designed to run under systemd (RestartOnFailure).
    """

    def __init__(self):
        self._running = False
        self._audio_q: queue.Queue = queue.Queue(maxsize=10)
        self._infer_q: queue.Queue = queue.Queue(maxsize=5)
        self._streak  = 0

        self._validate_paths()
        self._load_models()

        with _suppress_stderr():
            self._pa  = pyaudio.PyAudio()
        self._mic = INMP441Reader(self._pa)

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_paths() -> None:
        missing = False
        for label, path in [
            ("Model file",       AudioConfig.MODEL_PATH),
            ("Class names file", AudioConfig.CLASS_NAMES_PATH),
        ]:
            if not Path(path).exists():
                log.error(f"[Startup] {label} not found: {path}")
                missing = True
        if missing:
            sys.exit(1)
        log.info("[Startup] Model paths OK.")

    # ── Model loading ─────────────────────────────────────────────────

    def _load_models(self) -> None:
        log.info("[Audio] Loading YAMNet …")
        self._yamnet = hub.load(AudioConfig.YAMNET_PATH)
        log.info("[Audio] YAMNet ready.")

        log.info("[Audio] Loading custom classifier …")
        self._clf = tf.keras.models.load_model(AudioConfig.MODEL_PATH)

        with open(AudioConfig.CLASS_NAMES_PATH) as fh:
            self.class_names = [l.strip() for l in fh]
        log.info(f"[Audio] Classes: {self.class_names}")

    # ── Thread: Mic capture ───────────────────────────────────────────

    def _audio_thread(self) -> None:
        log.info("[AudioThread] Started.")
        n_chunks = int(
            AudioConfig.SAMPLE_RATE / AudioConfig.CHUNK * AudioConfig.RECORD_SECONDS
        )

        while self._running:
            try:
                audio = self._mic.read(n_chunks)
            except OSError as exc:
                if self._running:
                    log.error(f"[AudioThread] Read error: {exc}")
                continue

            # Silence gate — skip quiet frames
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms <= AudioConfig.RMS_SILENCE_GATE:
                continue

            # Peak-normalise with gain
            peak  = np.max(np.abs(audio)) + 1e-6
            normd = np.clip((audio / peak) * AudioConfig.INFERENCE_GAIN, -1.0, 1.0)

            try:
                self._audio_q.put_nowait(normd)
            except queue.Full:
                pass   # inference thread backed up; drop frame

        log.info("[AudioThread] Stopped.")

    # ── Thread: ML inference ─────────────────────────────────────────

    def _inference_thread(self) -> None:
        log.info("[InferenceThread] Started.")

        while self._running:
            try:
                audio_48k = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Downsample 48 kHz → 16 kHz for YAMNet
            audio_16k = librosa.resample(
                audio_48k,
                orig_sr=AudioConfig.SAMPLE_RATE,
                target_sr=AudioConfig.YAMNET_RATE,
            )

            # YAMNet → 1024-D embeddings
            _, embeddings, _ = self._yamnet(audio_16k)

            # Custom classifier → monkey probability
            prob = float(
                self._clf.predict(
                    embeddings.numpy()[0].reshape(1, -1), verbose=0
                )[0][0]
            )

            try:
                self._infer_q.put_nowait({"prob": prob})
            except queue.Full:
                pass

        log.info("[InferenceThread] Stopped.")

    # ── Thread: Decision ─────────────────────────────────────────────

    def _decision_thread(self) -> None:
        """
        Maintains a consecutive-hit streak.
        When streak reaches REQUIRED_STREAK, fires trigger_alert()
        and resets to avoid immediate re-triggering.
        """
        log.info("[DecisionThread] Started.")

        while self._running:
            try:
                r = self._infer_q.get(timeout=1.0)
            except queue.Empty:
                continue

            prob   = r["prob"]
            ml_hit = prob >= AudioConfig.DETECTION_THRESHOLD
            self._streak = (self._streak + 1) if ml_hit else 0

            log.debug(
                f"[Decision] prob={prob:.3f}  "
                f"hit={'YES' if ml_hit else 'no '}  streak={self._streak}"
            )

            if self._streak >= AudioConfig.REQUIRED_STREAK:
                self._streak = 0
                confidence   = round(prob * 100, 2)
                log.warning(f"[Decision] MONKEY DETECTED — {confidence:.1f}% confidence")
                trigger_alert(source="audio", confidence=confidence)

        log.info("[DecisionThread] Stopped.")

    # ── Run / Shutdown ────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        threads = [
            threading.Thread(target=self._audio_thread,     name="AudioThread",     daemon=True),
            threading.Thread(target=self._inference_thread, name="InferenceThread", daemon=True),
            threading.Thread(target=self._decision_thread,  name="DecisionThread",  daemon=True),
        ]
        for t in threads:
            t.start()

        log.info("\n" + "=" * 56)
        log.info("  MONKEY AUDIO DETECTOR")
        log.info("=" * 56)
        log.info(f"  Audio device    : {AudioConfig.AUDIO_DEVICE_INDEX}")
        log.info(f"  Sample rate     : {AudioConfig.SAMPLE_RATE} Hz")
        log.info(f"  ML threshold    : {AudioConfig.DETECTION_THRESHOLD * 100:.0f}%")
        log.info(f"  Streak needed   : {AudioConfig.REQUIRED_STREAK}")
        log.info(f"  Classes         : {self.class_names}")
        log.info(f"  Alert socket    : /tmp/monkey_alert.sock")
        log.info("=" * 56 + "\n")

        try:
            while True:
                import time; time.sleep(1)
        except KeyboardInterrupt:
            log.info("\n[System] Shutting down …")
        finally:
            self._running = False
            import time; time.sleep(0.5)
            self._mic.close()
            self._pa.terminate()
            log.info("[System] Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MonkeyAudioDetector().run()
