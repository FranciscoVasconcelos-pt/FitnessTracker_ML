# Live Tracker — Guia de uso

Tracker em tempo real com **sensores do telemóvel** + **página web** + **backend Python no PC** (mesma WiFi).

**Estado atual:** gravar sets no telemóvel para re-treinar o mesmo `exercise_classifier.pkl`.

Roadmap completo: [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md)

---

## O que é a “Opção C”?

Ao planear o ML live, considerámos três abordagens (detalhe em [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md#opções-para-ml-no-telemóvel-a-b-c)):

- **Opção A** — usar o modelo MetaMotion actual no telemóvel sem re-treinar (pipeline pesado no PC; pouco fiável no iPhone).
- **Opção B** — criar um modelo simples novo só para live (rápido, mas o teu modelo actual ficaria à margem).
- **Opção C** ✅ — **gravar exercícios com o telemóvel**, correr o **mesmo pipeline** que já tens (`remove_outliers` → `build_features` → `save_model.py`) e **substituir** `exercise_classifier.pkl` por uma versão treinada com os teus dados.

Em resumo: **não jogamos fora o modelo anterior** — usamo-lo como referência offline e re-treinamos a mesma arquitectura (Random Forest + features completas) com gravações do iPhone. Por isso o site tem “Record set” / “Save to PC” e o script `make_dataset_phone.py`.

---

## Pré-requisitos

| Requisito | Detalhe |
|---|---|
| PC com Python | Ambiente conda `tracking-barbell-exercises` |
| Telemóvel | iPhone (Safari) ou Android (Chrome) |
| Rede | PC e telemóvel na **mesma WiFi** |
| OpenSSL | Necessário para gerar certificado HTTPS (já incluído no Git for Windows ou instalável separadamente) |

---

## Instalação

### 1. Ativar o ambiente conda

```bash
conda activate tracking-barbell-exercises
cd MLFitnessTracker
```

### 2. Instalar dependências do servidor web

Se ainda não instalaste FastAPI e Uvicorn:

```bash
pip install fastapi "uvicorn[standard]"
```

Estas dependências também estão listadas no `environment.yml`.

---

## Como arrancar

### 1. Iniciar o servidor

Na raiz do projeto:

```bash
python src/app/main.py
```

(`serve_web.py` também funciona — é um alias.)

O terminal mostra algo como:

```
HTTPS is required for iPhone motion sensors.
Open on this PC:  https://127.0.0.1:8000
Open on iPhone:   https://192.168.1.95:8000  (use Safari)
On iPhone: accept the certificate warning, then tap Start.
Press Ctrl+C to stop.
```

Na primeira execução, o script gera automaticamente certificados em `certs/` (ignorados pelo Git).

### 2. Abrir no telemóvel

Usa o URL **HTTPS** que o script imprime — substitui pelo IP do teu PC:

```
https://192.168.1.95:8000
```

> **Importante:** usa `https://`, não `http://`. No iPhone, HTTP bloqueia os sensores de movimento.

### 3. iPhone (Safari)

1. Abre o URL no **Safari** (não Chrome — sensores limitados noutros browsers iOS)
2. Aparece aviso de certificado → **Mostrar detalhes** → **Visitar este website**
3. Toca **Start**
4. Quando pedido, toca **Permitir** (acesso a movimento e orientação)
5. Move o telemóvel — o gráfico e os valores acc/gyro devem atualizar

### 4. Android (Chrome)

1. Abre `https://<IP-do-PC>:8000` no Chrome
2. Aceita o aviso de certificado se aparecer
3. Toca **Start** e move o telemóvel

---

## O que deves ver

| Elemento | Descrição |
|---|---|
| **Sample rate** | Taxa de amostragem estimada (~30–60 Hz, depende do telemóvel) |
| **Buffer size** | Número de amostras no gráfico (máx. 120) |
| **Accelerometer** | `acc_x`, `acc_y`, `acc_z` em m/s² |
| **Gyroscope** | Rotação alpha / beta / gamma em deg/s |
| **Gráfico live** | Três linhas (X vermelho, Y verde, Z azul) |
| **Packets sent** | Quantos buffers enviados ao PC (~1.5 s cada) |
| **PC total samples** | Total de leituras recebidas pelo backend |
| **Last response** | `ok` se o PC respondeu ao último envio |

No terminal do PC deves ver linhas como:

```
POST /sensor: 45 readings (acc_z 9.81 → 10.12, total 450)
```

Experimenta posições diferentes (braço, bolso, mão) e anota qual dá sinal mais estável — isso será útil nas fases de gravação e re-treino.

---

## Fase 2 — fluxo de dados

```
iPhone sensores → devicemotion (local)
                      ↓
                 sendBuffer (JS)
                      ↓  fetch POST ~1.5 s
                 PC FastAPI /sensor
                      ↓
                 {"status": "ok", "received": N, ...}
```

Formato enviado pelo telemóvel:

```json
{
  "readings": [
    {
      "t": 1730000000123,
      "acc_x": 0.12,
      "acc_y": -0.05,
      "acc_z": 9.81,
      "gyro_x": 1.2,
      "gyro_y": 0.0,
      "gyro_z": -0.3
    }
  ]
}
```

Para hosting cloud (Fase 6), edita `API_URL` no topo de `web/app.js`:

```javascript
const API_URL = "https://your-app.onrender.com";
```

---

## Gravar e re-treinar (Opção C)

O modelo MetaMotion (~99%) mantém-se para offline/baseline (`predict_model.py`). Para o telemóvel, gravamos **os teus** dados e re-treinamos o **mesmo** pipeline — ver secção [O que é a Opção C?](#o-que-é-a-opção-c) acima.

### Gravar um set no iPhone

1. **Start** (tracking activo)
2. Escolhe exercício + categoria (heavy / medium)
3. **Record set** — faz o exercício (5 ou 10 reps)
4. **Save to PC** — CSV guardado em `data/raw/phone/`

No terminal do PC:

```
POST /record: saved 312 samples → user-squat-heavy-20260909_154500.csv
```

### Processar e re-treinar (PC)

```bash
conda activate tracking-barbell-exercises
cd MLFitnessTracker

# 1. Ingestão phone → pkl @ 5 Hz
python src/data/make_dataset_phone.py

# 2. Copiar para o nome que o pipeline espera (ou editar os scripts)
copy data\interim\01_data_processed_phone.pkl data\interim\01_data_processed.pkl

# 3. Pipeline normal
python src/features/remove_outliers.py
python src/features/build_features.py
python src/models/save_model.py
```

O `exercise_classifier.pkl` fica re-treinado com dados do telemóvel. Depois ligamos `/predict` live (próximo passo).

**Meta de gravação:** 5–10 sets por exercício, **sempre a mesma posição** do telemóvel.

---

## Parar o servidor

No terminal onde o servidor está a correr:

```
Ctrl+C
```

---

## Resolução de problemas

### Porta 8000 já em uso

```
ERROR: [Errno 10048] ... bind on address ('0.0.0.0', 8000)
```

Outra instância do servidor ainda está ativa. Liberta a porta:

```powershell
netstat -ano | findstr :8000
taskkill /PID <numero_do_PID> /F
```

Depois volta a correr `python src/app/serve_web.py`.

### iPhone: "DeviceMotion is not supported"

Causas habituais:

| Causa | Solução |
|---|---|
| URL em HTTP | Usa `https://` |
| Browser errado | Usa **Safari** |
| Certificado não aceite | Aceita o aviso antes de tocar Start |

### Permissão negada

Toca **Start** outra vez e escolhe **Permitir**. Se negaste antes, pode ser preciso ir a **Definições → Safari → Movimento e orientação** e reativar.

### Telemóvel não abre a página

- Confirma que PC e telemóvel estão na mesma WiFi
- Verifica se o **Firewall do Windows** permite ligações entrantes na porta 8000
- Confirma o IP do PC: `ipconfig` → adaptador WiFi → IPv4

### Erro ao gerar certificado (OpenSSL)

O script precisa do comando `openssl` no PATH. Instala [Git for Windows](https://git-scm.com/) (inclui OpenSSL) ou OpenSSL standalone. Depois apaga `certs/` e volta a correr o servidor.

---

## Estrutura de ficheiros

```
web/
├── index.html      # UI mobile
├── app.js          # DeviceMotion + gráfico
└── style.css       # Estilos

src/app/
└── serve_web.py    # FastAPI + HTTPS + servir ficheiros estáticos

certs/              # Gerado localmente (não vai para o GitHub)
├── key.pem         # Chave privada
├── cert.pem        # Certificado público
└── openssl.cnf     # Config OpenSSL
```

---

## Hosting vs local — gravação de treino

No **PC local** (`python src/app/main.py`), a gravação para treino fica **activa** por defeito — CSVs vão para `data/raw/phone/`.

No **host cloud** (Render, Railway, etc.), define:

```bash
ENABLE_RECORDING=false
```

| Onde | `ENABLE_RECORDING` | UI “Record training set” | `POST /record` |
|------|--------------------|--------------------------|----------------|
| PC local | `true` (default) | Visível | Guarda CSV |
| Cloud | `false` | Escondida | 403 Forbidden |

Assim o modelo público serve só **predict**; os teus dados curados ficam no PC e não misturam com uploads de outros.

---

## Segurança e Git

A pasta `certs/` está no `.gitignore`. **Não commits** `key.pem` nem `cert.pem` — são gerados automaticamente em cada máquina. Os certificados são auto-assinados só para desenvolvimento local; no ginásio (Fase 6) usar-se-á HTTPS real via hosting cloud.

---

## Próximos passos (Opção C)

| Passo | Objetivo |
|---|---|
| **Gravar sets** | 5–10 por exercício via Record set / Save to PC |
| **Pipeline + save_model** | Re-treinar `exercise_classifier.pkl` |
| **/predict live** | Mostrar exercício previsto no telemóvel |
| **Reps + hosting** | Fases finais |

Ver [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md).
