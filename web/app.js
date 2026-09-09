// Config
const MAX_SAMPLES = 120;       // chart window (last N points)
const CHART_PADDING = 24;
const SEND_INTERVAL_MS = 1500; // how often we POST to the PC; will decrease for better accuracy in the future

// Empty = same server as the page. Set a full URL when we deploy to the cloud.
const API_URL = "";

// --- State ---
const state = {
  active: false,
  samples: [],
  sampleTimes: [],
  sendBuffer: [],
  sendTimer: null,
  lastSampleMs: null,
  estimatedHz: 0,
  packetsSent: 0,
  lastBackendStatus: null,
};

const elements = {
  startBtn: document.getElementById("start-btn"),
  stopBtn: document.getElementById("stop-btn"),
  status: document.getElementById("status"),
  accX: document.getElementById("acc-x"),
  accY: document.getElementById("acc-y"),
  accZ: document.getElementById("acc-z"),
  gyroX: document.getElementById("gyro-x"),
  gyroY: document.getElementById("gyro-y"),
  gyroZ: document.getElementById("gyro-z"),
  sampleRate: document.getElementById("sample-rate"),
  sampleCount: document.getElementById("sample-count"),
  packetsSent: document.getElementById("packets-sent"),
  backendTotal: document.getElementById("backend-total"),
  backendStatus: document.getElementById("backend-status"),
  canvas: document.getElementById("motion-chart"),
};

const ctx = elements.canvas.getContext("2d");

function apiBase() {
  return API_URL || window.location.origin;
}

function setStatus(message, type = "info") {
  elements.status.textContent = message;
  elements.status.dataset.type = type;
}

function updateBackendUI(data) {
  elements.packetsSent.textContent = String(state.packetsSent);
  elements.backendTotal.textContent = String(data?.total_samples ?? "—");
  elements.backendStatus.textContent = data?.status ?? "—";
  elements.backendStatus.dataset.type = data?.status === "ok" ? "success" : "error";
}

function resizeCanvas() {
  const rect = elements.canvas.getBoundingClientRect();
  elements.canvas.width = rect.width * window.devicePixelRatio;
  elements.canvas.height = rect.height * window.devicePixelRatio;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  drawChart();
}

function formatValue(value) {
  return value == null ? "—" : value.toFixed(2);
}

function updateReadouts(event) {
  const acc = event.accelerationIncludingGravity;
  const gyro = event.rotationRate;

  elements.accX.textContent = formatValue(acc?.x);
  elements.accY.textContent = formatValue(acc?.y);
  elements.accZ.textContent = formatValue(acc?.z);
  elements.gyroX.textContent = formatValue(gyro?.alpha);
  elements.gyroY.textContent = formatValue(gyro?.beta);
  elements.gyroZ.textContent = formatValue(gyro?.gamma);
}

function updateSampleRate(now) {
  if (state.lastSampleMs != null) {
    const delta = now - state.lastSampleMs;
    if (delta > 0) {
      const instantHz = 1000 / delta;
      state.estimatedHz = state.estimatedHz
        ? state.estimatedHz * 0.9 + instantHz * 0.1
        : instantHz;
    }
  }
  state.lastSampleMs = now;
  elements.sampleRate.textContent = `${state.estimatedHz.toFixed(1)} Hz`;
}

// Accumulate readings for the next POST /sensor (separate from the chart buffer).
function storeReading(event) {
  const acc = event.accelerationIncludingGravity;
  const gyro = event.rotationRate;
  if (!acc || acc.x == null) {
    return;
  }

  state.sendBuffer.push({
    t: Date.now(),
    acc_x: acc.x,
    acc_y: acc.y,
    acc_z: acc.z,
    gyro_x: gyro?.alpha ?? null,
    gyro_y: gyro?.beta ?? null,
    gyro_z: gyro?.gamma ?? null,
  });
}

// Feed the on-screen chart only — runs at full sensor rate (~30–60 Hz).
function pushSample(event) {
  const acc = event.accelerationIncludingGravity;
  if (!acc || acc.x == null) {
    return;
  }

  const now = performance.now();
  updateSampleRate(now);

  state.samples.push({ x: acc.x, y: acc.y, z: acc.z });
  state.sampleTimes.push(now);

  if (state.samples.length > MAX_SAMPLES) {
    state.samples.shift();
    state.sampleTimes.shift();
  }

  elements.sampleCount.textContent = String(state.samples.length);
  drawChart();
}

// Drain sendBuffer and ship it to FastAPI. Called on a timer and once on Stop.
async function flushSendBuffer() {
  if (state.sendBuffer.length === 0) {
    return;
  }

  const readings = state.sendBuffer.splice(0);

  try {
    const response = await fetch(`${apiBase()}/sensor`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ readings }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    state.packetsSent += 1;
    state.lastBackendStatus = data;
    updateBackendUI(data);
  } catch (error) {
    elements.backendStatus.textContent = "error";
    elements.backendStatus.dataset.type = "error";
    setStatus(`Backend error: ${error.message}`, "error");
  }
}

function startSendLoop() {
  stopSendLoop();
  state.sendTimer = setInterval(flushSendBuffer, SEND_INTERVAL_MS);
}

function stopSendLoop() {
  if (state.sendTimer != null) {
    clearInterval(state.sendTimer);
    state.sendTimer = null;
  }
}

function drawSeries(samples, key, color, min, max, width, height) {
  if (samples.length < 2) {
    return;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  samples.forEach((sample, index) => {
    const x = CHART_PADDING + (index / (MAX_SAMPLES - 1)) * (width - CHART_PADDING * 2);
    const normalized = (sample[key] - min) / (max - min || 1);
    const y = height - CHART_PADDING - normalized * (height - CHART_PADDING * 2);

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();
}

function drawChart() {
  const width = elements.canvas.clientWidth;
  const height = elements.canvas.clientHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, width, height);

  if (state.samples.length === 0) {
    ctx.fillStyle = "#9ca3af";
    ctx.font = "14px sans-serif";
    ctx.fillText("Waiting for sensor data...", CHART_PADDING, height / 2);
    return;
  }

  const values = state.samples.flatMap((sample) => [sample.x, sample.y, sample.z]);
  const min = Math.min(...values) - 1;
  const max = Math.max(...values) + 1;

  drawSeries(state.samples, "x", "#ef4444", min, max, width, height);
  drawSeries(state.samples, "y", "#22c55e", min, max, width, height);
  drawSeries(state.samples, "z", "#3b82f6", min, max, width, height);

  ctx.fillStyle = "#d1d5db";
  ctx.font = "12px sans-serif";
  ctx.fillText("X", width - 70, 20);
  ctx.fillStyle = "#ef4444";
  ctx.fillText("●", width - 55, 20);
  ctx.fillStyle = "#d1d5db";
  ctx.fillText("Y", width - 40, 20);
  ctx.fillStyle = "#22c55e";
  ctx.fillText("●", width - 25, 20);
  ctx.fillStyle = "#d1d5db";
  ctx.fillText("Z", width - 10, 20);
  ctx.fillStyle = "#3b82f6";
  ctx.fillText("●", width + 5, 20);
}

// Fired by iOS/Android whenever the IMU has a new sample.
function onDeviceMotion(event) {
  if (!state.active) {
    return;
  }
  updateReadouts(event);
  storeReading(event);
  pushSample(event);
}

// iOS needs HTTPS + Safari; Chrome on iPhone won't give us DeviceMotion.
function getMotionSupportError() {
  if (!window.isSecureContext) {
    return (
      "Motion sensors need HTTPS on iPhone. " +
      "Use https://<PC-IP>:8000 in Safari (not http://)."
    );
  }

  if (typeof DeviceMotionEvent === "undefined") {
    return "DeviceMotion is not available. On iPhone, use Safari (not Chrome).";
  }

  return null;
}

// Must run from a user  iOS — called inside startTracking().
async function requestMotionPermission() {
  if (typeof DeviceMotionEvent?.requestPermission === "function") {
    const response = await DeviceMotionEvent.requestPermission();
    if (response !== "granted") {
      throw new Error("Motion sensor permission denied.");
    }
  }

  if (typeof DeviceOrientationEvent?.requestPermission === "function") {
    const response = await DeviceOrientationEvent.requestPermission();
    if (response !== "granted") {
      throw new Error("Orientation sensor permission denied.");
    }
  }
}

// Start: ask permission, listen to devicemotion, begin periodic uploads.
async function startTracking() {
  const supportError = getMotionSupportError();
  if (supportError) {
    setStatus(supportError, "error");
    return;
  }

  try {
    await requestMotionPermission();
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  state.active = true;
  state.samples = [];
  state.sampleTimes = [];
  state.sendBuffer = [];
  state.lastSampleMs = null;
  state.estimatedHz = 0;
  state.packetsSent = 0;
  state.lastBackendStatus = null;

  window.addEventListener("devicemotion", onDeviceMotion);
  startSendLoop();
  elements.startBtn.disabled = true;
  elements.stopBtn.disabled = false;
  setStatus("Tracking active. Sending sensor data to PC every ~1.5 s.", "success");
  drawChart();
  updateBackendUI(null);
}

async function stopTracking() {
  state.active = false;
  window.removeEventListener("devicemotion", onDeviceMotion);
  stopSendLoop();
  await flushSendBuffer();
  elements.startBtn.disabled = false;
  elements.stopBtn.disabled = true;
  setStatus("Tracking stopped.", "info");
}

elements.startBtn.addEventListener("click", startTracking);
elements.stopBtn.addEventListener("click", stopTracking);
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
setStatus("Tap Start and allow motion access when prompted.", "info");
