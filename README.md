# ML Fitness Tracker

Projeto de aprendizagem em **Machine Learning aplicado a sensores wearables** para reconhecer exercícios de ginásio e contar repetições a partir de dados de acelerómetro e giroscópio.

O pipeline processa gravações do sensor **MetaMotion (MetaWear)**, extrai features temporais e de frequência, treina classificadores para identificar o exercício em curso e usa deteção de picos para estimar o número de reps por set.

---

## Objetivos

| Objetivo | Descrição |
|---|---|
| **Classificação de exercícios** | Prever qual exercício está a ser executado (`bench`, `squat`, `row`, `ohp`, `dead`, `rest`) |
| **Contagem de repetições** | Estimar reps por set com filtro passa-baixo + deteção de máximos locais |
| **Aprendizagem de conceitos** | Pipeline completo de data science: ingestão → limpeza → features → modelação → avaliação |

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
│   │   ├── train_model.py        # Treino e comparação de modelos
│   │   └── LearningAlgorithms.py # Wrapper dos classificadores sklearn
│   └── visualization/
│       └── plot_settings.py
├── environment.yml
└── README.md
```

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

Instalar dependências adicionais usadas no código (não incluídas no `environment.yml`):

```bash
pip install scikit-learn scipy seaborn
```

---

## Como executar o pipeline

Correr os scripts **por ordem**, a partir da pasta onde se encontra cada ficheiro (os paths relativos assumem execução a partir de `src/`):

```bash
# 1. Processar dados brutos → data/interim/01_data_processed.pkl
python src/data/make_dataset.py

# 2. (Opcional) Explorar visualmente os dados
python src/data/visualize.py

# 3. Remover outliers → data/interim/02_outliers_removed_chauvenet.pkl
python src/features/remove_outliers.py

# 4. Construir features → data/interim/03_data_features.pkl
python src/features/build_features.py

# 5. Treinar e comparar modelos
python src/models/train_model.py

# 6. (Opcional) Contar repetições por set
python src/features/count_repetitions.py
```

> **Nota:** Os scripts foram escritos como pipelines lineares (estilo notebook). Abrem janelas de gráficos com `plt.show()` durante a execução.

---

## Pipeline em detalhe

### 1. Ingestão (`make_dataset.py`)

- Lê todos os CSVs de acelerómetro e giroscópio
- Extrai metadados do nome do ficheiro (`participant`, `label`, `category`)
- Faz merge das leituras dos dois sensores
- Reamostra para **5 Hz** (intervalo de 200 ms)
- Exporta `01_data_processed.pkl`

### 2. Outliers (`remove_outliers.py`)

Compara três métodos de deteção:

- **IQR** — intervalo interquartil
- **Chauvenet** — critério estatístico (método escolhido para remoção)
- **LOF** — Local Outlier Factor

Outliers são substituídos por `NaN` (por label) e exportados em `02_outliers_removed_chauvenet.pkl`.

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

### 4. Modelação (`train_model.py`)

**Classificadores comparados:**

- Neural Network (MLP)
- Random Forest
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes

**Conjuntos de features testados:**

1. Sensores base (acc + gyro)
2. + magnitudes (`acc_r`, `gyro_r`)
3. + features temporais
4. + features de frequência + cluster
5. Subset selecionado por forward selection

**Avaliação:**

- Split aleatório estratificado (75/25)
- Split por participante (treino: B–E, teste: A) — avaliação mais realista para dados de sensores corporais

### 5. Contagem de reps (`count_repetitions.py`)

- Remove sets de `rest`
- Aplica filtro passa-baixo ao sinal
- Deteta picos com `scipy.signal.argrelextrema`
- Compara com ground truth (5 reps heavy / 10 reps medium)
- Calcula MAE (Mean Absolute Error)

---

## Resultados esperados

Os scripts geram gráficos interativos durante a execução:

- Boxplots e histogramas de outliers
- Elbow plot para escolha de k no K-Means
- Scatter 3D de clusters vs labels
- Bar plot comparando accuracy por modelo e feature set
- Matriz de confusão do melhor modelo
- Gráficos de contagem de reps por exercício

Figuras exportadas ficam em `reports/figures/` (via `visualize.py`).

---

## Limitações conhecidas

- **Merge acc/gyro:** o join atual é por posição de linha, não por timestamp alinhado — pode causar desalinhamento entre sensores
- **Data leakage:** o forward selection avalia no mesmo conjunto de treino; o split aleatório pode misturar janelas do mesmo set
- **Sem inferência em produção:** não existe script `predict_model.py` nem modelo serializado
- **Paths relativos:** os scripts usam `../../data/...` e dependem do diretório de execução
- **Ambiente incompleto:** `scikit-learn`, `scipy` e `seaborn` precisam de instalação manual

---

## Referências

Este projeto segue a estrutura [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) e baseia-se nos conceitos do livro:

> Mark Hoogendoorn & Burkhardt Funk — *Machine Learning for the Quantified Self* (Springer, 2017)

Material de apoio e curso: [Dave Ebbelaar — ML Fitness Tracker](https://www.youtube.com/watch?v=7t2alSnE2-I)

Classes utilitárias (`DataTransformation`, `TemporalAbstraction`, `FrequencyAbstraction`, `LearningAlgorithms`) derivam do repositório [ML4QS](https://github.com/mhoogen/ML4QS).

---

## Licença

Ver ficheiro `LICENSE` na raiz do repositório.
