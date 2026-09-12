# Live Tracker — To-Do List

Plano de implementação para **telefóvel + website + backend local** (hosting cloud depois).

Objetivo: ver o tracker a funcionar em **tempo real** sem MetaMotion, usando sensores do telemóvel e backend Python no PC (mesma WiFi).

> **Como usar:** ver [LIVE_TRACKER.md](LIVE_TRACKER.md)

---

## Opções para ML no telemóvel (A, B, C)

> **Nota:** isto é diferente da “Opção B” de **arquitetura** no README (telefóvel + web + PC *vs* sensor MetaMotion). Aqui falamos só de **como usar o modelo ML** no telemóvel.

Quando passámos da Fase 2 para ML live, havia três caminhos possíveis:

| Opção | Ideia | Prós | Contras |
|---|---|---|---|
| **A** | Usar o `exercise_classifier.pkl` **tal como está** no PC — replicar `build_features.py` (FFT, PCA, janelas) em tempo real sobre o buffer do telemóvel | Aproveita o modelo ~99% que já treinaste | Muito código no servidor; sensores MetaMotion ≠ iPhone → accuracy fraca na prática |
| **B** | Criar um **modelo novo e simples** só para live (RF com acc/gyro + `acc_r`/`gyro_r`, sem FFT/PCA) | Rápido de implementar; inferência leve | O trabalho do pipeline “grande” fica paralelo, não evolui; dois modelos diferentes |
| **C** ✅ | **Gravar dados com o telemóvel**, passar pelo **mesmo pipeline** (`make_dataset` → outliers → features → `save_model.py`) e **re-treinar** `exercise_classifier.pkl` | Reutilizas todo o projeto ML; um só modelo; adaptado ao iPhone | Precisas gravar sets em casa antes; demora mais até ter algo útil no ginásio |

**Escolhemos a Opção C** porque quiseste que o modelo que já treinaste **tivesse uso** — não um modelo descartável à parte, mas uma evolução treinada com **os teus** movimentos e **o teu** telemóvel.

---

## Estratégia escolhida: Opção C (resumo)

Reutilizar o **mesmo pipeline e o mesmo ficheiro `exercise_classifier.pkl`** — gravar no telemóvel, re-treinar, depois ligar `/predict` live.

```
Gravar sets no telemóvel → data/raw/phone/
        ↓
make_dataset_phone.py → 01_data_processed_phone.pkl
        ↓
remove_outliers → build_features → save_model.py
        ↓
models/exercise_classifier.pkl  (re-treinado com os teus dados)
        ↓
POST /predict live (Fase 3, depois do re-treino)
```

O modelo MetaMotion (~99%) continua como **baseline offline**; o modelo re-treinado passa a ser o usado no telemóvel.

---

## Fase 1 — Prova de sensores ✅

- [x] Página web + `DeviceMotionEvent` + gráfico live
- [x] HTTPS para iPhone

---

## Fase 2 — Backend + comunicação ✅

- [x] `src/app/main.py` + `POST /sensor`
- [x] Frontend envia buffer a cada ~1.5 s
- [x] Testado no telemóvel

---

## Fase 3 — Gravar dados + re-treino (Opção C) ✅

- [x] Modo “gravar” no site + `POST /record` → CSV em `data/raw/phone/`
- [x] `src/data/make_dataset_phone.py`
- [x] Pipeline + `save_model.py` (~93% holdout)
- [x] `POST /predict` + UI live + votação

---

## Fase 4 — Reps live — em pausa

- [x] Peak detection + contador no ecrã
- [ ] Afinar (falsos reps em descanso) — **depois do hosting**

---

## Fase 5 — Hosting — em curso

- [x] Modo produção (`PORT` / Docker, sem cert local)
- [x] `ENABLE_RECORDING=false` no cloud
- [x] `Dockerfile` + `render.yaml` + `requirements.txt`
- [ ] Push GitHub + deploy Render/Railway
- [ ] Testar no ginásio com URL HTTPS

---

## Ordem recomendada (Opção C)

```
Fase 1–2 ✅  →  Gravar sets  →  make_dataset_phone  →  pipeline  →  re-treino
                                                                    ↓
                                                          /predict live  →  reps  →  hosting
```

---

## Estrutura de pastas

```
MLFitnessTracker/
├── data/raw/phone/          # CSVs gravados pelo telemóvel
├── web/                     # UI + sensores + gravação
├── src/app/main.py          # FastAPI (/sensor, /record, /predict)
├── src/data/make_dataset_phone.py
├── models/exercise_classifier.pkl   # re-treinado com dados phone
└── docs/LIVE_TRACKER.md
```

---

## Próximo passo

1. Commit + push (inclui `models/*.pkl`, `Dockerfile`, `render.yaml`)
2. Render → Blueprint ou Web Service (Docker)
3. Abrir URL HTTPS no iPhone (Safari) e testar ML live
4. Reps: afinar depois
