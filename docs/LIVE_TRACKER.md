# Live Tracker — Guia de uso

Tracker em tempo real com **sensores do telemóvel** + **página web** + **backend Python no PC** (mesma WiFi).

**Estado atual:** Fase 2 — gráfico live + envio de buffers ao PC via `POST /sensor` (sem ML ainda).

Roadmap completo: [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md)

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

## Segurança e Git

A pasta `certs/` está no `.gitignore`. **Não commits** `key.pem` nem `cert.pem` — são gerados automaticamente em cada máquina. Os certificados são auto-assinados só para desenvolvimento local; no ginásio (Fase 6) usar-se-á HTTPS real via hosting cloud.

---

## Próximas fases

| Fase | Objetivo |
|---|---|
| **2** | Telemóvel envia buffers ao PC via `POST /sensor` |
| **3** | Modelo ML simplificado + previsão live de exercício |
| **4** | Gravar dados teus e re-treinar com telemóvel |
| **5** | Contagem de reps em tempo real |
| **6** | Deploy cloud (Render/Railway) para usar no ginásio |

Ver detalhes em [LIVE_TRACKER_TODO.md](LIVE_TRACKER_TODO.md).
