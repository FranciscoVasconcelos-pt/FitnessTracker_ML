"""FastAPI backend for the live fitness tracker.

Serves the web UI over HTTPS, receives sensor batches, and saves phone
recordings so we can retrain exercise_classifier.pkl on iPhone data.
"""

import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
PHONE_RAW = ROOT / "data" / "raw" / "phone"
CERT_DIR = ROOT / "certs"
KEY_FILE = CERT_DIR / "key.pem"
CERT_FILE = CERT_DIR / "cert.pem"

# Running totals for /health and /sensor responses (in-memory for now).
total_samples_received = 0
total_packets_received = 0

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "total_samples": total_samples_received,
        "total_packets": total_packets_received,
    }


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


# Save a full training set from the phone (one exercise = one CSV).
# Files land in data/raw/phone/ and are picked up by make_dataset_phone.py.
@app.post("/record", response_model=RecordResponse)
def save_recording(payload: RecordPayload):
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
    # 0.0.0.0 so the phone on the same WiFi can reach server
    host = "0.0.0.0"
    port = 8000
    ip = local_ip()
    ensure_dev_cert(ip)

    print("HTTPS is required for iPhone motion sensors.")
    print(f"Open on this PC:  https://127.0.0.1:{port}")
    print(f"Open on iPhone:   https://{ip}:{port}  (use Safari on ios)")
    print("Phone → POST /sensor (live) and POST /record (training sets).")
    print(f"Recordings saved to: {PHONE_RAW}")
    print("On iPhone: accept the certificate warning, then tap Start.")
    print("Press Ctrl+C to stop.")
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_keyfile=str(KEY_FILE),
        ssl_certfile=str(CERT_FILE),
    )


if __name__ == "__main__":
    main()
