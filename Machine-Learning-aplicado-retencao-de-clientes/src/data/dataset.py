from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

REQUIRED_TARGET_COLUMN = "churn"

# Common aliases from Portuguese and English datasets.
COLUMN_ALIASES: Dict[str, str] = {
    "cliente_id": "customer_id",
    "id_cliente": "customer_id",
    "customerid": "customer_id",
    "id": "customer_id",
    "cancelou": "churn",
    "churned": "churn",
    "is_churn": "churn",
    "dias_recencia": "recency_days",
    "recencia_dias": "recency_days",
    "recency": "recency_days",
    "frequencia_90d": "purchase_frequency_90d",
    "compras_90d": "purchase_frequency_90d",
    "ticket_medio": "avg_ticket",
    "valor_medio": "avg_ticket",
    "tickets_suporte_90d": "support_tickets_90d",
    "chamados_90d": "support_tickets_90d",
    "atraso_pagamento_dias": "payment_delay_days",
    "inadimplencia_dias": "payment_delay_days",
    "falhas_pagamento_90d": "failed_payments_90d",
    "login_dias_30d": "login_days_30d",
    "dias_ativos_30d": "login_days_30d",
    "engajamento_30d": "engagement_30d",
    "uso_ratio": "usage_ratio",
    "nps": "nps_score",
    "satisfacao": "satisfaction_score",
    "plano_tipo": "plan_type",
    "tipo_plano": "plan_type",
    "tipo_contrato": "contract_type",
    "metodo_pagamento": "payment_method",
    "regiao": "region",
    "valor_plano": "plan_value",
    "tempo_base_meses": "tenure_months",
}


def _normalize_column_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    return COLUMN_ALIASES.get(normalized, normalized)


def _coerce_churn_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    positive_tokens = {"1", "true", "yes", "sim", "cancelou", "churn"}
    return series.astype(str).str.strip().str.lower().isin(positive_tokens).astype(int)


def load_dataset(csv_path: str | Path, require_target: bool = True) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={col: _normalize_column_name(col) for col in df.columns})

    if "churn" in df.columns:
        df["churn"] = _coerce_churn_column(df["churn"])

    if require_target and REQUIRED_TARGET_COLUMN not in df.columns:
        raise ValueError(
            "O dataset de treino precisa conter a coluna alvo 'churn' "
            "(ou alias como 'cancelou', 'is_churn')."
        )

    if "customer_id" not in df.columns:
        df["customer_id"] = pd.RangeIndex(start=1, stop=len(df) + 1)

    return df
