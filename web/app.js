// Config
const MAX_SAMPLES = 120;       // chart window (last N points)
const CHART_PADDING = 24;
const SEND_INTERVAL_MS = 250;  // max gap between POST /predict (live feel)
const MIN_SEND_INTERVAL_MS = 200;
const SEND_BATCH_SIZE = 10;    // also send when this many samples are queued
const VOTE_REQUIRED = 3;       // consecutive matching predictions before UI updates
const CONFIDENCE_MIN = 0.55;   // only show exercise above this confidence

// Empty = same server as the page. Set a full URL when we deploy to the cloud.
const API_URL = "";

// --- State ---
const state = {
  active: false,
  samples: [],
  sampleTimes: [],
  sendBuffer: [],
  recordBuffer: [],
  recording: false,
  sendTimer: null,
  lastSampleMs: null,
  estimatedHz: 0,
  packetsSent: 0,
  lastBackendStatus: null,
  lastPrediction: null,
  lastSendAt: 0,
  predictRequestSeq: 0,
  lastAppliedSeq: 0,
  predictInFlight: false,
  voteBuffer: [],
  displayedExercise: null,
  displayedConfidence: null,
  displayedReps: 0,
  repTarget: 5,
  repExercise: "squat",
  repCountingActive: false,
  confirmedExerciseLabel: null,
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
  predictExercise: document.getElementById("predict-exercise"),
  predictConfidence: document.getElementById("predict-confidence"),
  predictStatus: document.getElementById("predict-status"),
  predictReps: document.getElementById("predict-reps"),
  predictRepTarget: document.getElementById("predict-rep-target"),
  repCountingStatus: document.getElementById("rep-counting-status"),
  repExerciseSelect: document.getElementById("rep-exercise"),
  repTargetSelect: document.getElementById("rep-target"),
  resetRepsBtn: document.getElementById("reset-reps-btn"),
  endSetBtn: document.getElementById("end-set-btn"),
  recordLabel: document.getElementById("record-label"),
  recordCategory: document.getElementById("record-category"),
  recordParticipant: document.getElementById("record-participant"),
  recordStartBtn: document.getElementById("record-start-btn"),
  recordSaveBtn: document.getElementById("record-save-btn"),
  recordCount: document.getElementById("record-count"),
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

function resetVoteState() {
  state.voteBuffer = [];
  state.displayedExercise = null;
  state.displayedConfidence = null;
  state.displayedReps = 0;
  state.confirmedExerciseLabel = null;
}

function updateRepsUI(data) {
  const target = data?.rep_target ?? state.repTarget;
  if (data?.reps != null) {
    state.displayedReps = data.reps;
  }
  if (data?.counting != null) {
    state.repCountingActive = data.counting;
  }

  elements.predictReps.textContent = String(state.displayedReps);
  elements.predictRepTarget.textContent = `/ ${target}`;
  elements.predictRepTarget.dataset.complete =
    state.repCountingActive && state.displayedReps >= target ? "true" : "false";
  if (elements.predictRepTarget.dataset.complete === "true") {
    elements.predictRepTarget.style.color = "#4ade80";
  } else {
    elements.predictRepTarget.style.color = "";
  }

  if (data?.set_complete) {
    state.repCountingActive = false;
    elements.repCountingStatus.textContent = `Set complete — ${state.displayedReps} reps`;
    elements.repCountingStatus.dataset.type = "success";
  } else if (state.repCountingActive) {
    const name = data?.locked_exercise_name || state.repExercise;
    elements.repCountingStatus.textContent = `Counting: ${name}`;
    elements.repCountingStatus.dataset.type = "success";
  } else {
    elements.repCountingStatus.textContent = "Tap Start set before reps";
    elements.repCountingStatus.dataset.type = "info";
  }
}

async function syncRepTarget(target) {
  state.repTarget = target;
  try {
    await fetch(`${apiBase()}/predict/rep-target`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
  } catch {
    // non-fatal
  }
}

async function startSet() {
  const exercise = elements.repExerciseSelect.value;
  state.repExercise = exercise;
  try {
    const response = await fetch(`${apiBase()}/predict/reset-reps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exercise }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    state.displayedReps = 0;
    state.repCountingActive = true;
    updateRepsUI({
      reps: 0,
      rep_target: state.repTarget,
      counting: true,
      locked_exercise: exercise,
      locked_exercise_name: data.locked_exercise_name,
    });
    elements.endSetBtn.disabled = false;
    setStatus(`Set started — counting ${data.locked_exercise_name || exercise}`, "success");
  } catch (error) {
    setStatus(`Start set failed: ${error.message}`, "error");
  }
}

async function endSet() {
  try {
    const response = await fetch(`${apiBase()}/predict/end-set`, { method: "POST" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    state.repCountingActive = false;
    updateRepsUI({
      reps: data.reps ?? state.displayedReps,
      rep_target: state.repTarget,
      counting: false,
      set_complete: true,
    });
    setStatus(`Set done — ${data.reps ?? state.displayedReps} reps`, "success");
  } catch (error) {
    setStatus(`End set failed: ${error.message}`, "error");
  }
}

function streakFromEnd(label) {
  let streak = 0;
  for (let i = state.voteBuffer.length - 1; i >= 0; i -= 1) {
    if (state.voteBuffer[i].label !== label) {
      break;
    }
    streak += 1;
  }
  return streak;
}

function recordVote(data) {
  const label = data.exercise;
  const name = data.exercise_name || label;
  const confidence = data.confidence ?? 0;

  if (confidence <= CONFIDENCE_MIN) {
    return { label, name, streak: 0, confirmed: false, lowConfidence: true, confidence };
  }

  state.voteBuffer.push({ label, name, confidence });
  if (state.voteBuffer.length > VOTE_REQUIRED * 2) {
    state.voteBuffer.shift();
  }

  const streak = streakFromEnd(label);
  const confirmed = streak >= VOTE_REQUIRED;
  if (confirmed) {
    const recent = state.voteBuffer.slice(-VOTE_REQUIRED);
    const avgConfidence =
      recent.reduce((sum, vote) => sum + vote.confidence, 0) / recent.length;
    if (avgConfidence > CONFIDENCE_MIN) {
      state.confirmedExerciseLabel = label;
      state.displayedExercise = name;
      state.displayedConfidence = avgConfidence;
    }
  }

  return { label, name, streak, confirmed, lowConfidence: false, confidence };
}

function updatePredictionUI(data) {
  if (!data) {
    resetVoteState();
    elements.predictExercise.textContent = "—";
    elements.predictConfidence.textContent = "—";
    elements.predictStatus.textContent = "Waiting for data…";
    elements.predictStatus.dataset.type = "info";
    updateRepsUI(null);
    return;
  }

  if (data.status === "warming_up") {
    elements.predictExercise.textContent = "…";
    elements.predictConfidence.textContent = "—";
    const progress =
      data.warmup_progress != null ? ` (${data.warmup_progress}%)` : "";
    elements.predictStatus.textContent =
      (data.message || "Collecting sensor data…") + progress;
    elements.predictStatus.dataset.type = "info";
    updateRepsUI(data);
    return;
  }

  if (data.status === "ok") {
    const vote = recordVote(data);
    const showExercise =
      state.displayedExercise &&
      state.displayedConfidence != null &&
      state.displayedConfidence > CONFIDENCE_MIN;

    if (showExercise) {
      elements.predictExercise.textContent = state.displayedExercise;
      elements.predictConfidence.textContent = `${Math.round(state.displayedConfidence * 100)}%`;
    } else {
      elements.predictExercise.textContent = "Incerto";
      elements.predictConfidence.textContent =
        vote.confidence > 0 ? `${Math.round(vote.confidence * 100)}%` : "—";
    }

    const pendingChange =
      !vote.lowConfidence &&
      vote.streak > 0 &&
      vote.streak < VOTE_REQUIRED &&
      (!state.displayedExercise || vote.name !== state.displayedExercise);

    if (vote.lowConfidence) {
      elements.predictStatus.textContent = "Confiança baixa (<55%)";
      elements.predictStatus.dataset.type = "info";
    } else if (pendingChange) {
      elements.predictStatus.textContent = `A confirmar ${vote.name} (${vote.streak}/${VOTE_REQUIRED})…`;
      elements.predictStatus.dataset.type = "info";
    } else if (showExercise) {
      elements.predictStatus.textContent = "Live prediction";
      elements.predictStatus.dataset.type = "success";
    } else {
      elements.predictStatus.textContent = "À espera de confiança >55%";
      elements.predictStatus.dataset.type = "info";
    }

    updateRepsUI(data);
  }
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

  const reading = {
    t: Date.now(),
    acc_x: acc.x,
    acc_y: acc.y,
    acc_z: acc.z,
    gyro_x: gyro?.alpha ?? null,
    gyro_y: gyro?.beta ?? null,
    gyro_z: gyro?.gamma ?? null,
  };

  state.sendBuffer.push(reading);

  if (state.recording) {
    state.recordBuffer.push(reading);
    elements.recordCount.textContent = `${state.recordBuffer.length} samples`;
  }

  maybeSendPredict();
}

function maybeSendPredict() {
  if (!state.active || state.sendBuffer.length === 0) {
    return;
  }

  const now = Date.now();
  const elapsed = now - state.lastSendAt;
  if (elapsed < MIN_SEND_INTERVAL_MS) {
    return;
  }
  if (state.sendBuffer.length >= SEND_BATCH_SIZE || elapsed >= SEND_INTERVAL_MS) {
    flushSendBuffer();
  }
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

// One predict in flight at a time; apply responses in order (ignore stale ones).
async function flushSendBuffer() {
  if (state.sendBuffer.length === 0 || state.predictInFlight) {
    return;
  }

  const readings = state.sendBuffer.splice(0);
  state.lastSendAt = Date.now();
  state.predictInFlight = true;
  const seq = ++state.predictRequestSeq;

  try {
    const response = await fetch(`${apiBase()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ readings }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    if (seq <= state.lastAppliedSeq) {
      return;
    }

    state.lastAppliedSeq = seq;
    state.packetsSent += 1;
    state.lastBackendStatus = data;
    state.lastPrediction = data;
    updateBackendUI(data);
    updatePredictionUI(data);
  } catch (error) {
    if (seq > state.lastAppliedSeq) {
      elements.backendStatus.textContent = "error";
      elements.backendStatus.dataset.type = "error";
      elements.predictStatus.textContent = "Backend error";
      elements.predictStatus.dataset.type = "error";
      setStatus(`Backend error: ${error.message}`, "error");
    }
  } finally {
    state.predictInFlight = false;
    if (state.active && state.sendBuffer.length > 0) {
      flushSendBuffer();
    }
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
  state.lastPrediction = null;
  state.lastSendAt = 0;
  state.predictRequestSeq = 0;
  state.lastAppliedSeq = 0;
  state.predictInFlight = false;
  resetVoteState();

  window.addEventListener("devicemotion", onDeviceMotion);
  startSendLoop();
  elements.startBtn.disabled = true;
  elements.stopBtn.disabled = false;
  elements.recordStartBtn.disabled = false;
  elements.resetRepsBtn.disabled = false;
  elements.endSetBtn.disabled = true;
  setStatus("Tracking active. Live predictions ~4×/s after warm-up.", "success");
  drawChart();
  updateBackendUI(null);
  updatePredictionUI(null);
  updateRepsUI({ reps: 0, rep_target: state.repTarget, counting: false });

  fetch(`${apiBase()}/predict/reset`, { method: "POST" }).catch(() => {});
  syncRepTarget(Number(elements.repTargetSelect.value));
}

async function stopTracking() {
  if (state.recording) {
    setStatus("Stop recording or save the set before stopping tracking.", "error");
    return;
  }

  state.active = false;
  window.removeEventListener("devicemotion", onDeviceMotion);
  stopSendLoop();
  await flushSendBuffer();
  elements.startBtn.disabled = false;
  elements.stopBtn.disabled = true;
  elements.recordStartBtn.disabled = true;
  elements.resetRepsBtn.disabled = true;
  elements.endSetBtn.disabled = true;
  setStatus("Tracking stopped.", "info");
}

function startRecording() {
  if (!state.active) {
    setStatus("Start tracking first, then record a set.", "error");
    return;
  }

  state.recording = true;
  state.recordBuffer = [];
  elements.recordStartBtn.disabled = true;
  elements.recordSaveBtn.disabled = false;
  elements.recordCount.textContent = "0 samples";
  setStatus(`Recording ${elements.recordLabel.value}… do the set, then Save.`, "success");
}

async function saveRecording() {
  if (!state.recording || state.recordBuffer.length === 0) {
    setStatus("Nothing to save — record a set first.", "error");
    return;
  }

  const payload = {
    participant: elements.recordParticipant.value.trim() || "user",
    label: elements.recordLabel.value,
    category: elements.recordCategory.value,
    readings: state.recordBuffer,
  };

  try {
    const response = await fetch(`${apiBase()}/record`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    state.recording = false;
    state.recordBuffer = [];
    elements.recordStartBtn.disabled = false;
    elements.recordSaveBtn.disabled = true;
    elements.recordCount.textContent = "0 samples";
    setStatus(`Saved ${data.samples} samples → ${data.filename}`, "success");
  } catch (error) {
    setStatus(`Save failed: ${error.message}`, "error");
  }
}

elements.startBtn.addEventListener("click", startTracking);
elements.stopBtn.addEventListener("click", stopTracking);
elements.recordStartBtn.addEventListener("click", startRecording);
elements.recordSaveBtn.addEventListener("click", saveRecording);
elements.resetRepsBtn.addEventListener("click", startSet);
elements.endSetBtn.addEventListener("click", endSet);
elements.repTargetSelect.addEventListener("change", () => {
  syncRepTarget(Number(elements.repTargetSelect.value));
  updateRepsUI({ reps: state.displayedReps, rep_target: state.repTarget, counting: true });
});
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
setStatus("Tap Start and allow motion access when prompted.", "info");
