"""Serve the Phase 1 live sensor web page on the local network."""

import os
import subprocess
import socket
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
CERT_DIR = ROOT / "certs"
KEY_FILE = CERT_DIR / "key.pem"
CERT_FILE = CERT_DIR / "cert.pem"

app = FastAPI(title="ML Fitness Tracker Live Sensors")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
def app_js():
    return FileResponse(WEB_DIR / "app.js")


@app.get("/style.css")
def style_css():
    return FileResponse(WEB_DIR / "style.css")


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
    print(f"Open on iPhone:   https://{ip}:{port}  (use Safari)")
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
