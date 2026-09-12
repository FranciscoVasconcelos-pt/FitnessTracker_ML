# Live Tracker — Guia de uso



Tracker em tempo real: **sensores do telemóvel** → **página web** → **backend Python** (PC local ou Render).



**URL live:** [https://ml-fitness-tracker.onrender.com](https://ml-fitness-tracker.onrender.com)  

**Roadmap:** [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md)



---



## Estado actual



| Componente | Estado |

|------------|--------|

| Sensores + gráficos acc/gyro | Feito |

| ML live (~93% holdout phone) | Feito |

| Hosting Render + HTTPS | Feito |

| Gravação para treino | Feito (só PC local) |

| Contagem de reps | Parcial — afinação pendente |



---



## Modo local (PC + WiFi)



### Arrancar



```bash

conda activate tracking-barbell-exercises

cd MLFitnessTracker

python src/app/main.py

```



Abre no iPhone (**Safari**): `https://<IP-do-PC>:8000`  

Aceita o certificado auto-assinado → **Start** → permitir motion.



### Modo cloud (Render)



Abre [https://ml-fitness-tracker.onrender.com](https://ml-fitness-tracker.onrender.com) no Safari.  

Não precisas de PC ligado; precisas de WiFi ou dados móveis.



**Free tier:** primeira visita após inactividade pode demorar ~30–60 s (cold start).



---



## UI — o que vês



| Secção | Descrição |

|--------|-----------|

| **Start / Stop** | Liga sensores e envio ao servidor |

| **Stats** | Sample rate, buffer, pacotes enviados |

| **Exercise** | Exercício detectado pelo ML (com votação) |

| **Reps** | Contador manual por exercício (Start set) |

| **Sensors** | Valores acc/gyro em tempo real |

| **Gráficos** | Aceleração + giroscópio (últimas ~120 amostras) |

| **Record** | Só visível no PC local — grava CSV para treino |



---



## Fluxo de dados (actual)



```

iPhone DeviceMotion

       ↓

  web/app.js

       ↓ POST /readings  (~8×/s, upload rápido)

       ↓ POST /predict   (~3×/s, ML + reps)

  FastAPI (main.py)

       ↓

  LivePredictor → exercise_classifier.pkl

```



Formato de cada leitura:



```json

{

  "t": 1730000000123,

  "acc_x": 0.12,

  "acc_y": -0.05,

  "acc_z": 9.81,

  "gyro_x": 1.2,

  "gyro_y": 0.0,

  "gyro_z": -0.3

}

```



`API_URL` em `web/app.js` fica `""` (same origin) — funciona igual em local e Render.



---



## ML live — como interpretar



1. **Warm-up ~5 s** após Start — buffer a encher

2. **Votação** — 3 previsões seguidas iguais para confirmar exercício

3. **Rest** — limiar mais alto (evita bloquear em descanso antes do set)

4. Move o braço **logo após** aquecer para não confirmar Rest parado



---



## Reps (Start set)



1. Escolhe exercício no dropdown (independente do ML)

2. **Start set** — começa a contar picos no acelerómetro

3. **End set** — para manualmente



A contagem usa `rep_counting.py` (lowpass + picos + gate de movimento/amplitude). Validar no ginásio — ver [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md#fase-4--reps-live).



---



## Gravar e re-treinar (Opção C — só local)



1. `python src/app/main.py` no PC

2. **Start** → **Record** → fazer set → **Save**

3. CSV em `data/raw/phone/`



Pipeline:



```bash

python src/data/make_dataset_phone.py

copy data\interim\01_data_processed_phone.pkl data\interim\01_data_processed.pkl

python src/features/remove_outliers.py

python src/features/build_features.py

python src/models/save_model.py

git add models/*.pkl && git commit -m "Retrain" && git push

```



Render faz redeploy automático com o modelo novo.



**Meta:** 5–10 sets por exercício, mesma posição do telemóvel, sets ≥15 s.



---



## Local vs cloud



| | PC local | Render |

|--|----------|--------|

| URL | `https://<IP-PC>:8000` | https://ml-fitness-tracker.onrender.com |

| Gravação treino | Sim | Não (desactivada) |

| HTTPS | Certificado local | Automático |

| Latência ML | Baixa | ~0,3–1 s por `/predict` |



Variável `ENABLE_RECORDING`: activa por defeito local; `false` no Docker/Render.



---



## Deploy / actualizar Render



1. Push para GitHub (`main`)

2. Render rebuild automático (~5–15 min)

3. Testar `/health` → `"predictor_ready": true`



Ficheiros: `Dockerfile`, `render.yaml`, `requirements.txt`.



---



## Resolução de problemas



### iPhone: sensores não funcionam

- Usa **Safari** e **HTTPS**

- Aceita certificado (local) ou abre URL Render



### ML preso em Rest

- Move-te logo após warm-up

- Confirma que o modelo está actualizado no Render



### Porta 8000 ocupada (local)



```powershell

netstat -ano | findstr :8000

taskkill /PID <PID> /F

```



### Cold start lento (Render)

- Espera ~1 min na 1.ª visita, ou usa [UptimeRobot](https://uptimerobot.com) em `/health`



---



## Estrutura



```

web/           index.html, app.js, style.css

src/app/       main.py, predict_live.py, live_features.py, live_reps.py
src/features/  rep_counting.py (lógica partilhada de reps)

models/        exercise_classifier.pkl, live_artifact.pkl

data/raw/phone/  CSVs gravados (local)

docs/          LIVE_TRACKER.md, LIVE_TRACKER_TODO.md

Dockerfile     deploy Render

```



---



## Segurança



`certs/` está no `.gitignore` — nunca commits chaves privadas.  

Gravação de treino desactivada no host público para não misturar dados de terceiros com os teus CSVs curados.

