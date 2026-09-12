"""FastAPI backend for the live fitness tracker.

Serves the web UI over HTTPS, receives sensor batches, runs live ML
predictions, and saves phone recordings for retraining.
"""

import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from predict_live import EXERCISE_NAMES, LivePredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
PHONE_RAW = ROOT / "data" / "raw" / "phone"
CERT_DIR = ROOT / "certs"
KEY_FILE = CERT_DIR / "key.pem"
CERT_FILE = CERT_DIR / "cert.pem"

# Running totals for /health and /sensor responses (in-memory for now).
total_samples_received = 0
total_packets_received = 0
live_predictor: Optional[LivePredictor] = None

# Terminal log throttling — avoid flooding on every /predict packet.
_log_state = {
    "exercise": None,
    "last_log_at": 0.0,
    "warmup_milestone": -1,
}
LOG_HEARTBEAT_SEC = 8.0
CONFIDENCE_MIN = 0.55  # match web/app.js — ignore low-confidence predictions


def is_production_host() -> bool:
    """True on Render/Railway/etc. where the platform terminates HTTPS."""
    if os.environ.get("PORT"):
        return True
    return bool(os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT"))


def recording_enabled() -> bool:
    """Local dev: on by default. Off on cloud unless ENABLE_RECORDING=true."""
    value = os.environ.get("ENABLE_RECORDING")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return not is_production_host()


app = FastAPI(title="ML Fitness Tracker Live")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# Shape of the JSON the phone sends every time stamp.
# Pydantic rejects bad payloads before our handler runs.
class Reading(BaseModel):
    t: float = Field(..., description="Unix timestamp in milliseconds")
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None


class SensorPayload(BaseModel):
    readings: List[Reading]


class SensorResponse(BaseModel):
    status: str
    received: int
    total_samples: int
    total_packets: int


class RecordPayload(BaseModel):
    participant: str = "user"
    label: str
    category: str = "heavy"
    readings: List[Reading]


class RecordResponse(BaseModel):
    status: str
    filename: str
    samples: int
    path: str


class PredictResponse(BaseModel):
    status: str
    exercise: Optional[str] = None
    exercise_name: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    warmup_progress: Optional[int] = None
    buffer_span_ms: Optional[int] = None
    samples: Optional[int] = None
    buffer_samples: Optional[int] = None
    resampled_rows: Optional[int] = None
    reps: Optional[int] = None
    rep_target: Optional[int] = None
    counting: Optional[bool] = None
    locked_exercise: Optional[str] = None
    locked_exercise_name: Optional[str] = None
    set_complete: Optional[bool] = None
    total_samples: int
    total_packets: int


class RepTargetPayload(BaseModel):
    target: int = Field(5, ge=1, le=20)


class LockExercisePayload(BaseModel):
    exercise: str


class ResetRepsPayload(BaseModel):
    exercise: Optional[str] = None


# Static files — the phone loads these once, then runs JS locally.
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
def app_js():
    return FileResponse(WEB_DIR / "app.js")


@app.get("/style.css")
def style_css():
    return FileResponse(WEB_DIR / "style.css")


def get_live_predictor():
    global live_predictor
    if live_predictor is None:
        live_predictor = LivePredictor()
    return live_predictor


def reset_log_state():
    global _log_state
    _log_state = {"exercise": None, "last_log_at": 0.0, "warmup_milestone": -1}


def log_prediction(result):
    """Print only on exercise change, warmup milestones, or occasional heartbeat."""
    global _log_state
    now = time.monotonic()

    if result["status"] == "warming_up":
        progress = result.get("warmup_progress") or 0
        milestone = max(m for m in (0, 25, 50, 75, 100) if progress >= m)
        if milestone > _log_state["warmup_milestone"]:
            print(f"[live] A aquecer… {progress}%")
            _log_state["warmup_milestone"] = milestone
        return

    confidence = result.get("confidence") or 0.0
    if confidence <= CONFIDENCE_MIN:
        if _log_state["exercise"] is not None:
            print(f"[live] -- Incerto ({confidence:.0%})")
            _log_state["exercise"] = None
            _log_state["last_log_at"] = now
        return

    exercise = result.get("exercise_name") or result.get("exercise", "?")
    changed = exercise != _log_state["exercise"]
    heartbeat = now - _log_state["last_log_at"] >= LOG_HEARTBEAT_SEC

    if changed:
        print(f"[live] >> {exercise} ({confidence:.0%})")
        _log_state["exercise"] = exercise
        _log_state["last_log_at"] = now
        _log_state["warmup_milestone"] = -1
    elif heartbeat:
        print(f"[live]    {exercise} ({confidence:.0%})")
        _log_state["last_log_at"] = now


@app.get("/health")
def health():
    predictor_ready = (ROOT / "models" / "live_artifact.pkl").exists()
    return {
        "status": "ok",
        "total_samples": total_samples_received,
        "total_packets": total_packets_received,
        "predictor_ready": predictor_ready,
        "recording_enabled": recording_enabled(),
    }


@app.get("/config")
def app_config():
    return {"recording_enabled": recording_enabled()}


# Main data path: phone POSTs a batch of acc/gyro readings collected since last send.
# We don't persist yet — just count and echo back so the UI can confirm delivery.
@app.post("/sensor", response_model=SensorResponse)
def receive_sensor(payload: SensorPayload):
    global total_samples_received, total_packets_received

    count = len(payload.readings)
    total_samples_received += count
    total_packets_received += 1

    if count > 0:
        first = payload.readings[0]
        last = payload.readings[-1]
        print(
            f"POST /sensor: {count} readings "
            f"(Data example: acc_z {first.acc_z:.2f} → {last.acc_z:.2f}, "
            f"total of {total_samples_received} samples)"
        )

    return SensorResponse(
        status="ok",
        received=count,
        total_samples=total_samples_received,
        total_packets=total_packets_received,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_exercise(payload: SensorPayload):
    global total_samples_received, total_packets_received

    count = len(payload.readings)
    total_samples_received += count
    total_packets_received += 1

    try:
        predictor = get_live_predictor()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Predictor not ready: {exc}. Run save_model.py first.",
        ) from exc

    predictor.add_readings(payload.readings)
    result = predictor.predict()
    log_prediction(result)

    return PredictResponse(
        **result,
        total_samples=total_samples_received,
        total_packets=total_packets_received,
    )


@app.post("/predict/reset")
def reset_predictor():
    global live_predictor
    if live_predictor is not None:
        live_predictor.clear()
    reset_log_state()
    print("[live] Sessão reiniciada — à espera de predição.")
    return {"status": "ok"}


@app.post("/predict/reset-reps")
def reset_reps(payload: ResetRepsPayload = ResetRepsPayload()):
    predictor = get_live_predictor()
    predictor.reset_reps(payload.exercise)
    locked = predictor.locked_rep_exercise
    name = None
    if locked:
        name = EXERCISE_NAMES.get(locked, locked)
    print(f"[live] Reps reset — counting {name or '—'}")
    return {
        "status": "ok",
        "reps": 0,
        "counting": predictor.rep_counting_active,
        "locked_exercise": locked,
        "locked_exercise_name": name,
    }


@app.post("/predict/rep-target")
def set_rep_target(payload: RepTargetPayload):
    predictor = get_live_predictor()
    predictor.set_rep_target(payload.target)
    return {"status": "ok", "rep_target": predictor.rep_target}


@app.post("/predict/lock-exercise")
def lock_exercise(payload: LockExercisePayload):
    predictor = get_live_predictor()
    predictor.lock_exercise(payload.exercise)
    return {
        "status": "ok",
        "locked_exercise": predictor.locked_rep_exercise,
    }


@app.post("/predict/end-set")
def end_set():
    predictor = get_live_predictor()
    predictor.end_set()
    print(f"[live] Set complete — {predictor.rep_count} reps")
    return {
        "status": "ok",
        "reps": predictor.rep_count,
        "set_complete": True,
    }


# Save a full training set from the phone (one exercise = one CSV).
# Files land in data/raw/phone/ and are picked up by make_dataset_phone.py.
@app.post("/record", response_model=RecordResponse)
def save_recording(payload: RecordPayload):
    if not recording_enabled():
        raise HTTPException(
            status_code=403,
            detail="Recording is disabled on this server. Use your local PC to save training sets.",
        )
    if not payload.readings:
        raise HTTPException(status_code=400, detail="No readings in recording.")

    PHONE_RAW.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{payload.participant}-{payload.label}-{payload.category}-{timestamp}.csv"
    filepath = PHONE_RAW / filename

    rows = [
        {
            "epoch_ms": int(r.t),
            "acc_x": r.acc_x,
            "acc_y": r.acc_y,
            "acc_z": r.acc_z,
            "gyro_x": r.gyro_x,
            "gyro_y": r.gyro_y,
            "gyro_z": r.gyro_z,
            "participant": payload.participant,
            "label": payload.label,
            "category": payload.category,
        }
        for r in payload.readings
    ]
    pd.DataFrame(rows).to_csv(filepath, index=False)

    print(f"POST /record: saved {len(rows)} samples → {filepath}")
    return RecordResponse(
        status="ok",
        filename=filename,
        samples=len(rows),
        path=str(filepath.relative_to(ROOT)),
    )


def local_ip():
    # Trick to get the LAN IP without caring which interface WiFi uses.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def ensure_dev_cert(ip: str):
    # iOS won't expose motion sensors over plain HTTP on a local IP.
    # Regenerate the cert if the PC's WiFi address changed.
    CERT_DIR.mkdir(exist_ok=True)
    marker = CERT_DIR / ".ip"
    openssl_cnf = CERT_DIR / "openssl.cnf"

    if KEY_FILE.exists() and CERT_FILE.exists() and marker.read_text().strip() == ip:
        return

    cnf_text = f"""[ req ]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[ dn ]
CN = localhost

[ v3_req ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = {ip}
"""
    openssl_cnf.write_text(cnf_text)

    env = {**os.environ, "OPENSSL_CONF": str(openssl_cnf)}
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(KEY_FILE),
            "-out",
            str(CERT_FILE),
            "-days",
            "365",
            "-nodes",
            "-config",
            str(openssl_cnf),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    marker.write_text(ip)


def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))

    if is_production_host():
        print(f"Production server on port {port} (HTTPS via hosting platform).")
        print("Phone → POST /predict (live ML). Recording:", recording_enabled())
        uvicorn.run(app, host=host, port=port, access_log=False)
        return

    ip = local_ip()
    ensure_dev_cert(ip)

    print("HTTPS is required for iPhone motion sensors.")
    print(f"Open on this PC:  https://127.0.0.1:{port}")
    print(f"Open on iPhone:   https://{ip}:{port}  (use Safari on ios)")
    print("Phone → POST /predict (live ML), POST /sensor, POST /record (training).")
    if recording_enabled():
        print(f"Recordings saved to: {PHONE_RAW}")
    else:
        print("Recording disabled (ENABLE_RECORDING=false).")
    print("On iPhone: accept the certificate warning, then tap Start.")
    print("Press Ctrl+C to stop.")
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_keyfile=str(KEY_FILE),
        ssl_certfile=str(CERT_FILE),
        access_log=False,
    )


if __name__ == "__main__":
    main()
