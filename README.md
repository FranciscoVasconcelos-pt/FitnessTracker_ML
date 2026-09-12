# ML Fitness Tracker

Projeto de aprendizagem em **Machine Learning aplicado a sensores wearables** para reconhecer exercícios de ginásio e contar repetições a partir de dados de acelerómetro e giroscópio.

O pipeline processa gravações do sensor **MetaMotion (MetaWear)**, extrai features temporais e de frequência, treina um classificador Random Forest para identificar o exercício em curso e usa deteção de picos para estimar o número de reps por set.

---

## Live tracker (telemóvel)

**Abrir no telemóvel (Safari no iPhone):** [https://ml-fitness-tracker.onrender.com](https://ml-fitness-tracker.onrender.com)

1. Tap **Start** → permitir acesso aos sensores de movimento  
2. Faz o exercício — o ML mostra o exercício detectado em tempo real  
3. No plano free do Render, a **primeira visita** após inactividade pode demorar ~1 min a “acordar” o servidor  

Gravação de sets para treino: só no **PC local** (`python src/app/main.py`). Ver [docs/LIVE_TRACKER.md](docs/LIVE_TRACKER.md).

---

## Objetivos

| Objetivo | Descrição |
|---|---|
| **Classificação de exercícios** | Prever qual exercício está a ser executado (`bench`, `squat`, `row`, `ohp`, `dead`, `rest`) |
| **Contagem de repetições** | Estimar reps por set com filtro passa-baixo + deteção de máximos locais |
| **Aprendizagem de conceitos** | Pipeline completo de data science: ingestão → limpeza → features → modelação → inferência |

---

## Dataset

Os dados brutos estão em `data/raw/MetaMotion/MetaMotion/` e consistem em ficheiros CSV exportados do MetaMotion, com leituras de:

- **Acelerómetro** — 12.5 Hz
- **Giroscópio** — 25 Hz

### Convenção de nomes dos ficheiros

```
{participante}-{exercício}-{categoria}_MetaWear_{timestamp}_..._{sensor}.csv
```

**Exemplo:** `A-bench-heavy2-rpe8_MetaWear_..._Accelerometer_12.500Hz_1.4.4.csv`

| Campo | Valores | Descrição |
|---|---|---|
| `participant` | A, B, C, D, E | Identificador do participante |
| `label` | bench, squat, row, ohp, dead, rest | Tipo de exercício |
| `category` | heavy, medium | Intensidade / carga (5 reps heavy, 10 reps medium) |

---

## Estrutura do projeto

```
MLFitnessTracker/
├── data/
│   ├── raw/              # CSVs originais (MetaMotion)
│   └── interim/          # Artefactos intermédios (.pkl) gerados pelo pipeline
├── models/
│   └── exercise_classifier.pkl   # Modelo treinado (gerado por save_model.py)
├── reports/
│   └── figures/          # Gráficos exportados
├── references/
│   └── folder_structure.txt
├── src/
│   ├── data/
│   │   ├── make_dataset.py       # Ingestão, merge e reamostragem
│   │   └── visualize.py          # Exploração visual dos dados
│   ├── features/
│   │   ├── remove_outliers.py    # Deteção e remoção de outliers
│   │   ├── build_features.py     # Feature engineering
│   │   ├── count_repetitions.py  # Contagem de reps por set
│   │   ├── DataTransformation.py # Low-pass filter e PCA
│   │   ├── TemporalAbstraction.py
│   │   └── FrequencyAbstraction.py
│   ├── models/
│   │   ├── train_model.py        # Treino completo e comparação de modelos
│   │   ├── save_model.py         # Treino rápido + guardar modelo
│   │   ├── predict_model.py      # Inferência com modelo guardado
│   │   └── LearningAlgorithms.py # Wrapper dos classificadores sklearn
│   ├── app/
│   │   ├── main.py               # FastAPI: HTTPS + POST /sensor (Fase 2+)
│   │   └── serve_web.py          # Alias para main.py
│   └── visualization/
│       └── plot_settings.py
├── web/                          # Frontend live tracker (HTML/JS/CSS)
├── docs/
│   ├── LIVE_TRACKER.md           # Como usar o live tracker
│   └── LIVE_TRACKER_TODO.md      # Roadmap de implementação
├── environment.yml
└── README.md
```

---

## Escolha de tecnologias

### Pipeline ML (offline)

| Tecnologia | Papel | Porquê |
|---|---|---|
| **Python 3.8** | Linguagem base | Versão original do projeto; compatível com scikit-learn e o resto do stack |
| **pandas / NumPy** | Manipulação de dados | Padrão em data science; leitura de CSV, merge temporal, janelas deslizantes |
| **SciPy** | Processamento de sinal | Filtros Butterworth, deteção de picos (`argrelextrema`) para contagem de reps |
| **scikit-learn** | Modelação | Random Forest como classificador principal — robusto, interpretável, bom desempenho em features tabulares |
| **Matplotlib / Seaborn** | Visualização | Exploração de dados, outliers, matrizes de confusão |
| **Conda** | Ambiente | Reprodutibilidade via `environment.yml`; gestão simples de dependências científicas |
| **pickle (`.pkl`)** | Persistência | Guardar datasets intermédios e o modelo treinado sem dependências extra |

O pipeline segue o livro *Machine Learning for the Quantified Self* (Hoogendoorn & Funk): abstrações temporais, FFT, PCA e classificação sobre dados de sensores IMU.

### Live tracker (tempo real — Opção B)

Sem hardware MetaMotion, o tracker em tempo real usa **telemóvel + browser + backend local no PC**:

| Tecnologia | Papel | Porquê |
|---|---|---|
| **HTML / CSS / JavaScript (vanilla)** | Frontend mobile | Sem build tools nem app nativa; abre no browser do telemóvel; fácil de iterar |
| **`DeviceMotionEvent` API** | Sensores no browser | Acede ao acelerómetro e giroscópio do telemóvel sem App Store / Google Play |
| **Canvas API** | Gráfico live | Desenho leve de acc_x/y/z no ecrã, sem bibliotecas externas |
| **FastAPI** | Backend Python | API REST moderna; na Fase 2+ recebe buffers de sensores e corre inferência ML |
| **Uvicorn** | Servidor ASGI | Servir a página web e endpoints com baixa latência |
| **HTTPS (certificado auto-assinado)** | Ligação segura | **Obrigatório no iPhone/Safari** — a Apple só expõe sensores de movimento em contexto seguro (`https://` ou `localhost`) |
| **OpenSSL** | Certificados locais | Gera `certs/key.pem` e `certs/cert.pem` automaticamente ao arrancar o servidor |

**Porquê esta arquitetura?**

- **Sem MetaMotion:** o telemóvel substitui o sensor wearable; qualquer pessoa pode testar em casa.
- **Sem app nativa:** evita Swift/Kotlin, publicação em lojas e permissões complexas — o browser já tem acesso aos sensores (com HTTPS no iOS).
- **Backend no PC (mesma WiFi):** para desenvolvimento não é preciso hosting cloud; na Fase 6 migra-se para Render/Railway mudando só o `API_URL`.
- **FastAPI + Python:** reutiliza o mesmo ecossistema do pipeline ML (pandas, sklearn, scipy) para inferência live nas fases seguintes.

> Guia de instalação e uso: **[docs/LIVE_TRACKER.md](docs/LIVE_TRACKER.md)**  
> Estratégia ML no telemóvel (**Opção C** — gravar dados e re-treinar o mesmo modelo): **[docs/LIVE_TRACKER_TODO.md](docs/LIVE_TRACKER_TODO.md#opções-para-ml-no-telemóvel-a-b-c)**

---

## Setup

### Pré-requisitos

- [Conda](https://docs.conda.io/en/latest/miniconda.html) ou Miniconda
- Python 3.8+

### Ambiente virtual

```bash
conda env create -f environment.yml
conda activate tracking-barbell-exercises
```

Se o ambiente já existir e faltarem dependências:

```bash
conda env update -f environment.yml --prune
```

O `environment.yml` inclui `scipy`, `scikit-learn` e `seaborn`. Se o conda instalar `pandas 2.x` e surgirem warnings com sklearn, podes fixar:

```bash
conda install pandas=1.5.2 -n tracking-barbell-exercises
```

---

## Como executar o pipeline

Correr **por ordem**, a partir da **raiz do projeto**:

```bash
conda activate tracking-barbell-exercises
cd MLFitnessTracker

# 1. Processar dados brutos → data/interim/01_data_processed.pkl
python src/data/make_dataset.py

# 2. (Opcional) Explorar visualmente os dados
python src/data/visualize.py

# 3. Remover outliers → data/interim/02_outliers_removed_chauvenet.pkl
python src/features/remove_outliers.py

# 4. Construir features → data/interim/03_data_features.pkl
python src/features/build_features.py

# 5. Treinar e guardar modelo → models/exercise_classifier.pkl
python src/models/save_model.py

# 6. Inferência com o modelo guardado
python src/models/predict_model.py

# 7. Contagem de repetições (output no terminal, sem gráficos)
cd src/features
python count_repetitions.py
cd ../..
```

### Atalho: treino completo (opcional, demorado)

O `train_model.py` compara vários classificadores (NN, RF, KNN, DT, NB) e abre gráficos interativos. Demora **30+ minutos**.

```bash
python src/models/train_model.py
```

No final também guarda o modelo em `models/exercise_classifier.pkl`.

---

## Pipeline em detalhe

### 1. Ingestão (`make_dataset.py`)

- Lê todos os CSVs de acelerómetro e giroscópio
- Extrai metadados do nome do ficheiro (`participant`, `label`, `category`)
- Emparelha ficheiros acc/gyro da mesma gravação pelo prefixo do nome
- Deduplica exports repetidos (versões 1.4.4 vs 1.4.41)
- Reamostra cada sensor para **5 Hz** (200 ms) e faz join temporal
- Atribui um `set` único por gravação emparelhada (~82 sets)
- Exporta `01_data_processed.pkl`

### 2. Outliers (`remove_outliers.py`)

Compara três métodos de deteção:

- **IQR** — intervalo interquartil
- **Chauvenet** — critério estatístico (método escolhido para remoção)
- **LOF** — Local Outlier Factor

Outliers são substituídos por `NaN` (por label) e exportados em `02_outliers_removed_chauvenet.pkl`.

> Abre muitas janelas de gráficos — fecha cada uma para o script continuar.

### 3. Features (`build_features.py`)

| Etapa | Técnica | Output |
|---|---|---|
| Imputação | Interpolação linear | Valores em falta preenchidos |
| Filtro | Butterworth passa-baixo | Sinal suavizado |
| Redução dimensional | PCA (3 componentes) | `pca_1`, `pca_2`, `pca_3` |
| Magnitude | √(x² + y² + z²) | `acc_r`, `gyro_r` |
| Temporais | Média e std em janelas deslizantes | `*_temp_mean_*`, `*_temp_std_*` |
| Frequência | FFT em janelas | `*_freq_*`, `*_pse`, `*_freq_weighted` |
| Clustering | K-Means (k=5) | Coluna `cluster` |

Exporta `03_data_features.pkl`.

### 4. Modelação

#### `save_model.py` (recomendado)

- Treina **Random Forest** com `feature_set_4` (todas as features)
- Split por participante: treino B–E, teste A
- Guarda em `models/exercise_classifier.pkl` (~2–5 min)

#### `train_model.py` (exploração completa)

- Compara NN, RF, KNN, Decision Tree, Naive Bayes
- Testa 5 conjuntos de features
- Forward selection + grid search
- Matrizes de confusão e bar plots de accuracy

### 5. Inferência (`predict_model.py`)

- Carrega `models/exercise_classifier.pkl`
- Prevê exercícios para o participante A (dados de teste)
- Imprime accuracy geral, accuracy por exercício e tabela completa de previsões

### 6. Contagem de reps (`count_repetitions.py`)

- Remove sets de `rest`
- Aplica filtro passa-baixo ao sinal
- Deteta picos com `scipy.signal.argrelextrema`
- Compara com ground truth (5 reps heavy / 10 reps medium)
- Imprime **MAE** e tabela média por exercício/categoria (sem gráficos interativos)

> Correr a partir de `src/features/` por causa dos imports locais.

---

## Resultados obtidos

| Métrica | Valor |
|---|---|
| **Classificação (participante A)** | ~99% accuracy |
| **Contagem de reps (MAE)** | ~0.79 reps por set |

Accuracy por exercício (participante A):

| Exercício | Accuracy |
|---|---|
| Bench Press | 100% |
| Squat | 100% |
| Row | 100% |
| Deadlift | ~99% |
| Rest | ~99% |
| Overhead Press | ~98% |

---

## Artefactos gerados

| Ficheiro | Gerado por |
|---|---|
| `data/interim/01_data_processed.pkl` | `make_dataset.py` |
| `data/interim/02_outliers_removed_chauvenet.pkl` | `remove_outliers.py` |
| `data/interim/03_data_features.pkl` | `build_features.py` |
| `models/exercise_classifier.pkl` | `save_model.py` ou `train_model.py` |
| `reports/figures/*.png` | `visualize.py` |

---

## Limitações conhecidas

- **Data leakage:** o forward selection em `train_model.py` avalia no mesmo conjunto de treino; o split aleatório pode misturar janelas do mesmo set
- **Contagem de reps:** MAE de ~0.79 — row e ohp medium têm maior erro; `row` deveria usar `gyro_x` mas o loop usa `acc_r`
- **Scripts exploratórios:** `remove_outliers.py`, `build_features.py` e `train_model.py` abrem gráficos interativos; `count_repetitions.py` corre sem janelas

---

## Referências

Este projeto segue a estrutura [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) e baseia-se nos conceitos do livro:

> Mark Hoogendoorn & Burkhardt Funk — *Machine Learning for the Quantified Self* (Springer, 2017)

Material de apoio e curso: [Dave Ebbelaar — ML Fitness Tracker](https://www.youtube.com/watch?v=7t2alSnE2-I)

Classes utilitárias (`DataTransformation`, `TemporalAbstraction`, `FrequencyAbstraction`, `LearningAlgorithms`) derivam do repositório [ML4QS](https://github.com/mhoogen/ML4QS).

---

## Licença

Ver ficheiro `LICENSE` na raiz do repositório.
