# Machine Learning - Retenção de Clientes (Churn Prediction)

Sistema de previsão de churn para identificar clientes com alta probabilidade de cancelamento antes da perda.

## Objetivo

Gerar `scores de risco de churn` a partir de dados históricos de comportamento, consumo e relacionamento do cliente para:

- reduzir cancelamentos;
- aumentar lifetime value;
- priorizar ações de retenção com base em dados.

## Arquitetura da Solução

- `Modelo base global`: `XGBoost` para capturar padrões amplos de churn.
- `Modelo personalizado`: `SGDClassifier` (incremental) para adaptação contínua via `partial_fit`.
- `Ensemble`: combinação de scores (`0.7` base + `0.3` personalizado).
- `API`: FastAPI para predição em tempo real.
- `Dashboard`: Streamlit para monitoramento de risco e priorização de clientes.

## Estrutura do Projeto

```text
.
├── data/
│   ├── raw/                  # CSVs de entrada (treino, score, feedback)
│   ├── processed/            # Saídas com scores
│   └── external/
├── models/
│   └── churn_model.joblib    # Artefato salvo após treino
├── reports/
│   └── model_metrics.json    # Métricas de validação
└── src/
    ├── data/
    │   └── dataset.py        # Leitura e normalização de schema
    ├── features/
    │   └── build_features.py # Feature engineering de churn
    ├── models/
    │   ├── training.py       # Treino e seleção dos modelos
    │   ├── inference.py      # Scoring e ação recomendada
    │   └── incremental.py    # Atualização incremental (SGD)
    ├── api/
    │   └── app.py            # FastAPI
    ├── dashboard/
    │   └── app.py            # Streamlit
    └── main.py               # CLI principal
```

## Instalação

```bash
pip install -r requirements.txt
```

## Formato mínimo do dataset

Colunas aceitas (aliases PT/EN são normalizados automaticamente):

- `customer_id`
- `churn` (apenas para treino e update)
- `tenure_months`
- `recency_days`
- `purchase_frequency_90d`
- `avg_ticket`
- `support_tickets_90d`
- `payment_delay_days`
- `failed_payments_90d`
- `login_days_30d`
- `engagement_30d`
- `usage_ratio`
- `nps_score`
- `satisfaction_score`
- `plan_value`
- `plan_type`, `contract_type`, `payment_method`, `region`

## Como executar

### 0) Gerar dados de exemplo (opcional)

```bash
python -m src.data.generate_sample
```

### 1) Treino

```bash
python -m src.main train --train-path data/raw/churn_train.csv
```

Saídas:

- `models/churn_model.joblib`
- `reports/model_metrics.json`

### 2) Score em lote

```bash
python -m src.main score --input-path data/raw/churn_score.csv
```

Saída:

- `data/processed/churn_scores.csv` com:
  - `customer_id`
  - `churn_risk_score`
  - `risk_level` (`baixo`, `medio`, `alto`)
  - `retention_action`

### 3) Atualização incremental do modelo personalizado

```bash
python -m src.main update --labeled-path data/raw/churn_feedback.csv
```

### 4) Subir API

```bash
uvicorn src.api.app:app --reload
```

Endpoint principal: `POST /predict`

### 5) Abrir dashboard

```bash
streamlit run src/dashboard/app.py
```

## Indicadores de Churn considerados

O pipeline foi estruturado para priorizar sinais de risco típicos de churn:

- queda de engajamento e uso;
- aumento de recência (tempo sem compra/uso);
- falhas e atrasos de pagamento;
- aumento de tickets de suporte;
- baixa satisfação/NPS;
- relação fraca entre valor percebido e consumo.

## Equipe

- Matheus Felipe Soares Silva
- Rhuan Rassilam Souza Santos
