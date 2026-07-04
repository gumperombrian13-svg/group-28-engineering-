"""
GSM_handler.py — SIM800L GSM Controller
════════════════════════════════════════════════════════════════════════════
Shared by alert_service.py (the only file that should import this).

Sections:
  §A  Serial Setup & Modem Init
  §B  AT Command Engine
  §C  Time Sync  (AT+CCLK → Pi system clock)  [now works without sudo password]
  §D  SMS Send   (CMS ERROR handling, ESC cleanup, retry)
  §E  SMS Alert Dispatch  [personalised greetings added]
  §F  SMS Receive / Inbox
  §G  Calls — Outgoing (reserved)
  §H  Calls — Incoming (auto hang-up)
  §I  Background URC Listener Thread
"""

import re
import time
import logging
import threading
import subprocess
from datetime import datetime

import serial

log = logging.getLogger("MonkeyWarning")


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

class GSMConfig:
    GSM_PORT             = "/dev/serial0"
    GSM_BAUDRATE         = 9600
    GSM_READ_TIMEOUT     = 2
    GSM_INIT_DELAY       = 3.0
    GSM_RECONNECT_DELAY  = 10.0

    ALERT_PHONE_NUMBERS  = {
        "Desire": "+256704558166",
        "Flavius":  "+256761024424",
        "Andrew" : "+256776299376",
        "Brian": "+256701445136",
        "Edmon" : "+256750073443",
    }

    SMS_DONE_TOKENS      = ("+CMGS:", "+CMS ERROR:", "ERROR")
    SMS_PROMPT_TIMEOUT   = 5.0
    SMS_CONFIRM_TIMEOUT  = 10.0
    SMS_RETRY_DELAY      = 4.0
    SMS_INTER_DELAY      = 1.5

    INCOMING_AUTO_HANGUP = True


# ══════════════════════════════════════════════════════════════════════
# GSM CONTROLLER
# ══════════════════════════════════════════════════════════════════════

class GSMController:
    """
    SIM800L serial driver.
    Thread-safe: _serial_lock serialises every AT exchange.
    Auto-reconnects if the serial port drops.
    """

    # ── §A  Serial Setup & Modem Init ──────────────────────────────────

    def __init__(self):
        self._serial          = None
        self._running         = False
        self._serial_lock     = threading.Lock()
        self._listener_thread = None

        self._open_serial()
        self._init_modem()
        self._start_listener()

    def _open_serial(self) -> None:
        try:
            self._serial = serial.Serial(
                port     = GSMConfig.GSM_PORT,
                baudrate = GSMConfig.GSM_BAUDRATE,
                timeout  = GSMConfig.GSM_READ_TIMEOUT,
                rtscts   = False,
                dsrdtr   = False,
            )
            time.sleep(GSMConfig.GSM_INIT_DELAY)
            self._serial.reset_input_buffer()
            log.info(f"[GSM] {GSMConfig.GSM_PORT} @ {GSMConfig.GSM_BAUDRATE} baud")
        except serial.SerialException as exc:
            raise RuntimeError(f"[GSM] Cannot open port: {exc}") from exc

    def _init_modem(self) -> None:
        for attempt in range(5):
            if "OK" in self._at_locked("AT", delay=2):
                log.info("[GSM] Modem awake.")
                break
            log.warning(f"[GSM] Wake ping {attempt + 1}/5 …")
            time.sleep(2)
        else:
            log.warning("[GSM] Modem silent — continuing anyway.")

        for cmd, label in [
            ("ATE0",              "Echo off"),
            ("AT+CMGF=1",         "SMS text mode"),
            ("AT+CNMI=1,2,0,0,0", "Route SMS to serial"),
            ("AT+CLIP=1",         "Caller-ID on"),
        ]:
            resp   = self._at_locked(cmd, delay=2)
            if "OK" not in resp:
                time.sleep(2)
                resp = self._at_locked(cmd, delay=2)
            log.info(f"[GSM] {label}: {'OK' if 'OK' in resp else 'FAILED'}")

    def _reconnect(self) -> None:
        log.warning("[GSM] Lost connection — reconnecting …")
        while self._running:
            time.sleep(GSMConfig.GSM_RECONNECT_DELAY)
            try:
                if self._serial and self._serial.is_open:
                    self._serial.close()
                self._open_serial()
                self._init_modem()
                log.info("[GSM] Reconnected.")
                return
            except Exception as exc:
                log.error(f"[GSM] Reconnect failed: {exc}")

    def close(self) -> None:
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=GSMConfig.GSM_READ_TIMEOUT + 1)
        if self._serial and self._serial.is_open:
            self._serial.close()
        log.info("[GSM] Port closed.")

    # ── §B  AT Command Engine ──────────────────────────────────────────

    def _at_locked(self, cmd: str, delay: float = 1.0) -> str:
        with self._serial_lock:
            self._serial.reset_input_buffer()
            self._serial.write((cmd + "\r").encode())
            time.sleep(delay)
            return self._serial.read_all().decode(errors="ignore").strip()

    def _read_until_locked(self, targets: tuple, timeout: float) -> str:
        """Read inside an already-held lock until a target token appears."""
        buf      = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            waiting = self._serial.in_waiting
            chunk   = self._serial.read(waiting if waiting > 0 else 1)
            if chunk:
                buf += chunk.decode(errors="ignore")
                if any(t in buf for t in targets):
                    break
            else:
                time.sleep(2)
        return buf

    def _cancel_sms_transaction(self) -> None:
        self._serial.write(b"\x1B")
        time.sleep(2)
        self._serial.read_all()

    def is_ready(self) -> bool:
        return "OK" in self._at_locked("AT", delay=2)

    def check_signal(self) -> str:
        resp = self._at_locked("AT+CSQ")
        log.info(f"[GSM] Signal: {resp}")
        return resp

    # ── §C  Time Sync (no sudo password required) ─────────────────────

    def sync_time_from_gsm(self) -> bool:
        """
        Set system time from GSM network using AT+CCLK.
        Works without password prompt by:
          - trying `sudo -n` (passwordless sudo, requires sudoers rule)
          - falling back to direct date if running as root
          - otherwise logs a warning but does not crash
        For headless field deployment, either run the script as root
        (e.g., via systemd service with User=root) or add to sudoers:
            pi ALL=(ALL) NOPASSWD: /bin/date, /sbin/hwclock
        """
        resp  = self._at_locked("AT+CCLK?", delay=3)
        match = re.search(
            r'\+CCLK:\s*"(\d{2})/(\d{2})/(\d{2}),(\d{2}):(\d{2}):(\d{2})([+-]\d+)?',
            resp,
        )
        if not match:
            log.warning("[GSM] CCLK parse failed — time not synced.")
            return False

        yy, mo, dd, hh, mm, ss = match.groups()[:6]
        tz_raw = match.group(7)
        dt_str = f"{2000 + int(yy)}-{mo}-{dd} {hh}:{mm}:{ss}"

        try:
            # Try passwordless sudo first (most common on production Pis)
            subprocess.run(
                ["sudo", "-n", "date", "-s", dt_str],
                capture_output=True, text=True, timeout=5, check=False
            )
            # If that failed, try without sudo (works if EUID==0)
            if subprocess.run(["date", "-s", dt_str],
                              capture_output=True, timeout=5).returncode != 0:
                raise PermissionError("Need root to set date")

            subprocess.run(["sudo", "-n", "hwclock", "--systohc"],
                           capture_output=True, timeout=5, check=False)
            # If hwclock fails without sudo, try direct
            subprocess.run(["hwclock", "--systohc"],
                           capture_output=True, timeout=5, check=False)

            tz = f"  (UTC {int(tz_raw)/4:+.2f}h)" if tz_raw else ""
            log.info(f"[GSM] Clock → {dt_str}{tz}")
            return True

        except Exception as exc:
            log.error(f"[GSM] Time sync failed (need root or passwordless sudo): {exc}")
            return False

    # ── §D  SMS Send ──────────────────────────────────────────────────

    def _send_sms_once(self, number: str, message: str) -> tuple[bool, str]:
        got_prompt = False
        try:
            with self._serial_lock:
                self._serial.reset_input_buffer()
                self._serial.write(b"AT+CMGF=1\r")
                time.sleep(0.4)
                self._serial.read_all()

                self._serial.write(f'AT+CMGS="{number}"\r'.encode())
                prompt = self._read_until_locked(
                    (">", "+CMS ERROR:", "ERROR"),
                    GSMConfig.SMS_PROMPT_TIMEOUT,
                )
                if ">" not in prompt:
                    return False, self._extract_cms_error(prompt) or "no '>' prompt"

                got_prompt = True
                self._serial.write((message + "\x1A").encode())
                confirm = self._read_until_locked(
                    GSMConfig.SMS_DONE_TOKENS,
                    GSMConfig.SMS_CONFIRM_TIMEOUT,
                )

                if "+CMGS:" in confirm:
                    return True, "OK"

                self._cancel_sms_transaction()
                return False, self._extract_cms_error(confirm) or "no +CMGS confirm"

        except Exception as exc:
            if got_prompt:
                try:
                    with self._serial_lock:
                        self._cancel_sms_transaction()
                except Exception:
                    pass
            return False, str(exc)

    @staticmethod
    def _extract_cms_error(buf: str) -> str:
        m = re.search(r"\+CMS ERROR:\s*(.+)", buf)
        if m:
            return f"+CMS ERROR: {m.group(1).strip()}"
        return "ERROR (no detail)" if "ERROR" in buf else ""

    def send_sms(self, number: str, message: str, retries: int = 1) -> bool:
        log.info(f"[GSM][SMS] → {number}")
        for attempt in range(1 + retries):
            ok, reason = self._send_sms_once(number, message)
            if ok:
                log.info(f"[GSM][SMS] Sent OK → {number}")
                return True
            if attempt < retries:
                log.warning(
                    f"[GSM][SMS] Attempt {attempt+1} failed ({reason}) "
                    f"— retry in {GSMConfig.SMS_RETRY_DELAY}s"
                )
                time.sleep(GSMConfig.SMS_RETRY_DELAY)
            else:
                log.error(f"[GSM][SMS] FAILED → {number}  ({reason})")
        return False

    # ── §E  SMS Alert Dispatch with personalised greetings ───────────

    def _get_greeting(self) -> str:
        """Return time-appropriate greeting."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 18:
            return "Good afternoon"
        else:   # 18-23 and 0-4
            return "Good evening"

    def send_alert_to_all(self, message: str) -> None:
        """
        Send alert to every number in ALERT_PHONE_NUMBERS.
        Message is prefixed with personalised greeting and recipient's name.
        Example: "Good afternoon Joshua,\nMonkey detected at zone 3"
        """
        log.info(f"[GSM] Alerting {len(GSMConfig.ALERT_PHONE_NUMBERS)} recipient(s)")
        greeting = self._get_greeting()
        sent = 0
        for name, number in GSMConfig.ALERT_PHONE_NUMBERS.items():
            full_message = f"{greeting} {name},\n{message}"
            if self.send_sms(number, full_message):
                sent += 1
            time.sleep(GSMConfig.SMS_INTER_DELAY)
        log.info(f"[GSM] Alert done: {sent}/{len(GSMConfig.ALERT_PHONE_NUMBERS)} sent.")

    # ── §F  SMS Inbox ─────────────────────────────────────────────────

    def read_all_sms(self) -> list:
        resp     = self._at_locked('AT+CMGL="ALL"', delay=1.5)
        messages = []
        lines    = resp.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("+CMGL:"):
                parts  = line.split(",", 4)
                index  = parts[0].replace("+CMGL:", "").strip()
                status = parts[1].strip().strip('"') if len(parts) > 1 else ""
                sender = parts[2].strip().strip('"') if len(parts) > 2 else ""
                body   = lines[i+1].strip() if (i+1) < len(lines) else ""
                messages.append({"index": index, "status": status,
                                  "sender": sender, "body": body})
                i += 2
            else:
                i += 1
        log.info(f"[GSM][SMS] {len(messages)} message(s) on SIM.")
        return messages

    def delete_all_read_sms(self) -> bool:
        return "OK" in self._at_locked("AT+CMGD=1,1", delay=1.0)

    def _on_sms_received(self, sender: str, body: str) -> None:
        log.info(f"[GSM][SMS] From {sender}: '{body}'")
        upper = body.strip().upper()
        if upper == "STATUS":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.send_sms(sender, f"System OK. Monitoring active.\nTime: {now}")
        elif upper == "SILENCE":
            log.info("[GSM][SMS] Remote SILENCE received.")

    # ── §G  Outgoing Calls (reserved) ────────────────────────────────

    def make_call(self, number: str) -> bool:
        log.info(f"[GSM][Call] Dialling {number} …")
        resp = self._at_locked(f"ATD{number};", delay=1.0)
        if "ERROR" in resp:
            log.error(f"[GSM][Call] Failed: {repr(resp)}")
            return False
        log.info("[GSM][Call] Ringing …")
        return True

    def hangup(self) -> bool:
        ok = "OK" in self._at_locked("ATH", delay=0.5)
        log.info(f"[GSM][Call] Hangup: {'OK' if ok else 'FAILED'}")
        return ok

    # ── §H  Incoming Calls ────────────────────────────────────────────

    def _handle_incoming_call(self, clip_line: str) -> None:
        caller = "unknown"
        try:
            caller = clip_line.split('"')[1]
        except IndexError:
            pass
        log.info(f"[GSM][Call] Incoming from {caller} — hanging up.")
        time.sleep(0.3)
        self.hangup()

    # ── §I  URC Listener Thread ───────────────────────────────────────

    def _start_listener(self) -> None:
        self._running         = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name="GSM-Listener",
            daemon=True,
        )
        self._listener_thread.start()
        log.info("[GSM] Listener started.")

    def _listener_loop(self) -> None:
        ring_pending = False
        while self._running:
            try:
                with self._serial_lock:
                    raw = self._serial.readline()

                line = raw.decode(errors="ignore").strip()
                if not line:
                    continue
                if any(x in line for x in ("+CTZV:", "*PSUTTZ:", "DST:")):
                    continue

                log.debug(f"[GSM][URC] {line}")

                if line == "RING":
                    ring_pending = True

                elif line.startswith("+CLIP:"):
                    if ring_pending and GSMConfig.INCOMING_AUTO_HANGUP:
                        self._handle_incoming_call(line)
                    ring_pending = False

                elif line == "NO CARRIER":
                    ring_pending = False

                elif line.startswith("+CMT:"):
                    with self._serial_lock:
                        body_raw = self._serial.readline()
                    body = body_raw.decode(errors="ignore").strip()
                    try:
                        sender = line.split('"')[1]
                    except IndexError:
                        sender = "unknown"
                    self._on_sms_received(sender, body)

                elif line.startswith("+CMTI:"):
                    try:
                        idx  = int(line.split(",")[1].strip())
                        resp = self._at_locked(f"AT+CMGR={idx}", delay=0.8)
                        ls   = resp.splitlines()
                        for j, l in enumerate(ls):
                            if l.startswith("+CMGR:"):
                                parts  = l.split(",", 4)
                                sender = parts[1].strip().strip('"') if len(parts) > 1 else "unknown"
                                body   = ls[j+1].strip() if (j+1) < len(ls) else ""
                                self._on_sms_received(sender, body)
                                break
                    except (ValueError, IndexError) as exc:
                        log.warning(f"[GSM][SMS] CMTI parse error: {exc}")

            except serial.SerialException as exc:
                if self._running:
                    log.error(f"[GSM][Listener] Serial error: {exc}")
                    self._reconnect()
            except Exception as exc:
                if self._running:
                    log.error(f"[GSM][Listener] Error: {exc}")

        log.info("[GSM] Listener stopped.")
