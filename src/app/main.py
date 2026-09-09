"""FastAPI backend for the live fitness tracker (Phase 2+)."""

import os
import socket
import subprocess
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
CERT_DIR = ROOT / "certs"
KEY_FILE = CERT_DIR / "key.pem"
CERT_FILE = CERT_DIR / "cert.pem"

total_samples_received = 0
total_packets_received = 0

app = FastAPI(title="ML Fitness Tracker Live")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


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


def local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def ensure_dev_cert(ip: str):
    """Create a self-signed cert for local HTTPS (required by iOS motion sensors)."""
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
    port = 8000
    ip = local_ip()
    ensure_dev_cert(ip)

    print("HTTPS is required for iPhone motion sensors.")
    print(f"Open on this PC:  https://127.0.0.1:{port}")
    print(f"Open on iPhone:   https://{ip}:{port}  (use Safari on ios)")
    print("Phase 2: phone sends sensor buffers to POST /sensor every ~1.5 s.")
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
