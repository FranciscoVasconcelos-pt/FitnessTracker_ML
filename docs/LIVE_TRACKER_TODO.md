# Live Tracker — To-Do List

Plano de implementação para **telefóvel + website + backend local** (hosting cloud depois).

Objetivo: ver o tracker a funcionar em **tempo real** sem MetaMotion, usando sensores do telemóvel e backend Python no PC (mesma WiFi).

> **Como usar (Fase 1):** ver [LIVE_TRACKER.md](LIVE_TRACKER.md)

---

## Fase 1 — Prova de sensores (1–2 dias)

**Objetivo:** Confirmar que o telemóvel envia dados úteis.

- [x] Criar página web (`web/index.html` + JS) que lê `DeviceMotionEvent`
- [x] Mostrar gráfico live de `acc_x`, `acc_y`, `acc_z` no ecrã
- [ ] Testar no telemóvel (mesma WiFi que o PC)
- [ ] Experimentar posição: braço com faixa vs bolso vs mão
- [ ] Anotar taxa de amostragem real (~Hz)

**Como correr:**

```bash
pip install fastapi "uvicorn[standard]"
python src/app/serve_web.py
```

Abrir no telemóvel: `https://<IP-do-PC>:8000` (HTTPS — obrigatório no iPhone/Safari).

No iPhone: usar **Safari**, aceitar aviso de certificado, depois Start → permitir movimento.

**Entregável:** Site que mostra movimento em tempo real (sem ML ainda).

---

## Fase 2 — Backend + comunicação (2–3 dias)

**Objetivo:** Telefóvel envia dados → PC recebe.

- [x] Criar `src/app/main.py` com **FastAPI**
- [x] Endpoint `POST /sensor` — recebe JSON com buffer de leituras
- [x] Servir a página web estática (pasta `web/`)
- [x] Frontend envia buffer a cada ~1–2 s
- [x] `API_URL` configurável (`""` = same origin; cloud depois)
- [ ] Testar no telemóvel e ver logs `POST /sensor` no PC

**Como correr:**

```bash
python src/app/main.py
```

**Entregável:** Telefóvel → PC; PC responde `{"status": "ok", "received": N, ...}`.

---

## Fase 3 — ML em tempo real (3–5 dias)

**Objetivo:** Prever exercício a partir do buffer.

- [ ] Treinar modelo **simples** (RF com `acc_x/y/z`, `gyro_x/y/z`, `acc_r`, `gyro_r`)
- [ ] Script `save_model_live.py` — features leves, inferência rápida
- [ ] Endpoint `POST /predict` → `{ "exercise": "Squat", "confidence": 0.92 }`
- [ ] UI no telemóvel mostra exercício previsto
- [ ] Testar com **replay** de CSV do dataset (simular stream antes do telemóvel real)

**Entregável:** Classificação live (accuracy inicial pode ser baixa).

> **Nota:** O modelo atual (`exercise_classifier.pkl`) foi treinado com MetaMotion e features pesadas (FFT, PCA, janelas). Não usar tal como está para live no telemóvel.

---

## Fase 4 — Dados teus + re-treino (1–2 semanas)

**Objetivo:** Modelo adaptado ao telemóvel.

- [ ] Modo “gravar” no site — guarda sessão em CSV
- [ ] Gravar 5–10 sets por exercício (bench, squat, row, ohp, dead)
- [ ] Posição **sempre igual** (ex.: braço direito)
- [ ] Adaptar pipeline ou criar `make_dataset_phone.py`
- [ ] Re-treinar e substituir `models/exercise_classifier.pkl`

**Entregável:** Modelo treinado com **os teus** dados de telemóvel.

---

## Fase 5 — Reps live (opcional)

**Objetivo:** Contar repetições como `count_repetitions.py`.

- [ ] Peak detection no buffer (filtro passa-baixo + `argrelextrema`)
- [ ] Mostrar contador de reps no ecrã
- [ ] Ajustar `cutoff` por exercício

---

## Fase 6 — Hosting (quando quiseres usar no ginásio)

**Objetivo:** Funcionar sem PC em casa.

- [ ] Deploy FastAPI em Render / Railway / Fly.io
- [ ] HTTPS automático (necessário para sensores no iOS)
- [ ] Trocar `API_URL` para `https://teu-app.onrender.com`
- [ ] Testar no ginásio com 4G/WiFi

---

## Estrutura de pastas sugerida

```
MLFitnessTracker/
├── web/
│   ├── index.html      # UI + sensores
│   └── app.js          # DeviceMotion + fetch API
├── src/
│   └── app/
│       ├── main.py     # FastAPI
│       └── predict_live.py
├── models/
│   └── exercise_classifier_live.pkl
└── docs/
    └── LIVE_TRACKER_TODO.md   # este ficheiro
```

---

## Ordem recomendada

```
Fase 1 → Fase 2 → Fase 3 (com replay) → Fase 4 → Fase 5 → Fase 6
```

---

## Testar em casa vs hosting

| Cenário | Hosting necessário? |
|---|---|
| Testar em casa, PC ligado, mesma WiFi | **Não** |
| Usar no ginásio só com telemóvel | **Sim** |
| Partilhar demo online / portfolio | **Sim** |

**Sem hosting:** telemóvel acede ao IP local do PC (ex.: `http://192.168.1.50:8000`).

**Com hosting:** mesmo código; só muda o `API_URL` no frontend.

```javascript
// Configurável desde o início
const API_URL = "http://192.168.1.50:8000";  // depois: "https://teu-app.onrender.com"
```

---

## Arquitetura (Opção B)

```
Telemóvel (acelerómetro + giroscópio)
        ↓  WiFi / browser
Website (JavaScript lê sensores)
        ↓  envia janelas de dados
Backend Python (FastAPI no PC)
        ↓
Modelo ML → exercício + reps
        ↓
Ecrã do telemóvel
```

---

## Próximo passo

**Fase 1** — página web que mostra sensores do telemóvel em tempo real.
