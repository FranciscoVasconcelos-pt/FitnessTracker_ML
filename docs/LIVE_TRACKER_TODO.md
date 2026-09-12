# Live Tracker — To-Do List

Plano de implementação para **telefóvel + website + backend** (local + cloud).

**Guia de uso:** [LIVE_TRACKER.md](LIVE_TRACKER.md)  
**URL live:** [https://ml-fitness-tracker.onrender.com](https://ml-fitness-tracker.onrender.com)

---

## Estado geral (v1)

| Fase | Estado |
|------|--------|
| Fase 1 — Sensores + HTTPS | Feito |
| Fase 2 — Backend FastAPI | Feito |
| Fase 3 — Gravar + re-treinar (Opção C) | Feito (~93% holdout) |
| Fase 3b — ML live + votação | Feito |
| Fase 4 — Reps live | Parcial — módulo unificado; afinação no ginásio |
| Fase 5 — Hosting Render | Feito |

**Pendente para fechar v1:** validação no ginásio + (opcional) reps v2.

---

## Opção C (resumo)

Gravar sets no telemóvel → `data/raw/phone/` → pipeline → `exercise_classifier.pkl` → inferência live.

```
data/raw/phone/*.csv
        ↓
make_dataset_phone.py
        ↓
remove_outliers → build_features → save_model.py
        ↓
models/exercise_classifier.pkl + live_artifact.pkl
        ↓
POST /readings + POST /predict (live)
```

Gravação (`POST /record`) **só no PC local** — no Render fica desactivada.

---

## Fase 4 — Reps live

Módulo partilhado: `src/features/rep_counting.py` (usado por `live_reps.py`, `count_repetitions.py`, `predict_live.py`).

- [x] Peak detection + contador no ecrã
- [x] Start set / End set / target por exercício
- [x] Unificar lógica batch + live (`rep_counting.py`)
- [x] Gates live opcionais (desactivados — só gap entre reps; activar se falsos reps em descanso)
- [ ] Validar contagem no ginásio (CSVs phone: reps reais nem sempre = heavy/medium no nome)

---

## Fase 5 — Hosting 

- [x] Docker + `render.yaml` + `requirements.txt`
- [x] Deploy: https://ml-fitness-tracker.onrender.com
- [x] `ENABLE_RECORDING=false` no cloud
- [x] `POST /readings` (upload rápido) + `POST /predict` (ML)
- [x] Preload do modelo no arranque

---

## Endpoints actuais

| Endpoint | Uso |
|----------|-----|
| `GET /` | UI mobile |
| `GET /health` | Health check |
| `GET /config` | `{ recording_enabled }` |
| `POST /readings` | Append sensores (rápido) |
| `POST /predict` | ML + reps (se set activo) |
| `POST /predict/reset` | Limpa buffer ML |
| `POST /predict/reset-reps` | Start set |
| `POST /predict/end-set` | Termina set |
| `POST /record` | Grava CSV (só local) |

---

## Rep counting — arquitectura (implementado)

| Ficheiro | Papel |
|----------|-------|
| `src/features/rep_counting.py` | Config, picos, batch, `register_peaks` live |
| `src/app/live_reps.py` | Re-export para a app |
| `src/features/count_repetitions.py` | Benchmark MAE MetaMotion |
| `src/app/predict_live.py` | Estado `RepCounterState` + gates live |

### Referência rápida

### 1. Config única (`REP_CONFIG`)

```python
REP_CONFIG = {
    "bench": { "cutoff": 0.4, "prominence": 0.14, "min_gap_samples": 11, ... },
    "ohp":   { "peak_order": 9, "min_peak_gap_ms": 3000, ... },
    ...
}
```

`live_reps.py` e `count_repetitions.py` importam daqui — uma só fonte para afinar.

### 2. Pipeline de picos (funções puras)

```
df @ 5 Hz
  → acc_r + lowpass
  → argrelextrema
  → filter by min gap (samples)
  → filter by prominence
  → filter by min amplitude (opcional, v2)
  → list[(peak_ms, amplitude)]
```

Funções propostas:

- `prepare_filtered_signal(df, exercise)` → signal, timestamps
- `find_peak_candidates(signal, cfg)` → índices
- `peaks_from_df(df, exercise, after_ms=0)` → `[(ms, amp), ...]`

### 3. Modo batch

`count_reps_in_set(df, exercise)` → `int`  
Substitui o loop em `count_repetitions.py`; aplica regras de gap temporal entre picos (como live).

### 4. Modo live (`RepCounterState`)

Estado mínimo em `predict_live.py`:

```python
@dataclass
class RepCounterState:
    counted_peak_times: set
    session_max_amplitude: float
    rep_counting_started_ms: int
    ...
```

Função `register_peaks(peaks, state, cfg)`:

- Respeita `min_peak_gap_ms` entre reps
- Ignora picos antes de `rep_counting_started_ms`
- **v2:** amplitude mínima + ratio à melhor rep do set (evita fidget em descanso)
- **v2:** só começa a contar após detectar movimento real (acc std > limiar durante 1 s)

### 5. Fim de set

Manter `_is_resting_after_set()` — após último rep, se acc estável ~4.5 s → `set_complete`.

Afinar limiares `IDLE_ACTIVITY_STD` / `IDLE_ACTIVITY_RANGE` com CSVs de rest.

### Testes locais

```bash
python scripts/test_live_reps.py
python src/features/count_repetitions.py
```

### 7. O que **não** mudaria

- ML de classificação (separado de reps)
- Utilizador escolhe exercício no dropdown para reps (não depende do ML)
- UI Start set / End set

---

## Próximos passos

1. Validar ML live no ginásio (URL Render)
2. (v2) Implementar `rep_counting.py` unificado
3. (Opcional) Mais dados OHP/row se ML falhar no ginásio
4. (Opcional) UptimeRobot no `/health`
