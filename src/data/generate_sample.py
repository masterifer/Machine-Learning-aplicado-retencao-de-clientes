from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _build_dataframe(n_rows: int, with_target: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "tenure_months": rng.uniform(1, 60, n_rows).round(1),
            "recency_days": rng.integers(1, 120, n_rows),
            "purchase_frequency_90d": rng.integers(0, 20, n_rows),
            "avg_ticket": rng.uniform(40, 1200, n_rows).round(2),
            "support_tickets_90d": rng.integers(0, 8, n_rows),
            "payment_delay_days": rng.integers(0, 30, n_rows),
            "failed_payments_90d": rng.integers(0, 4, n_rows),
            "login_days_30d": rng.integers(0, 30, n_rows),
            "usage_ratio": rng.uniform(0, 1, n_rows).round(3),
            "nps_score": rng.integers(0, 11, n_rows),
            "satisfaction_score": rng.integers(1, 6, n_rows),
            "plan_value": rng.uniform(39.9, 399.9, n_rows).round(2),
            "plan_type": rng.choice(["basico", "padrao", "premium"], n_rows, p=[0.45, 0.4, 0.15]),
            "contract_type": rng.choice(["mensal", "anual"], n_rows, p=[0.65, 0.35]),
            "payment_method": rng.choice(["cartao", "boleto", "pix", "debito"], n_rows),
            "region": rng.choice(["sudeste", "sul", "nordeste", "norte", "centro-oeste"], n_rows),
        }
    )

    if with_target:
        # Synthetic churn logic to produce a realistic training signal.
        logit = (
            0.03 * df["recency_days"]
            + 0.2 * df["failed_payments_90d"]
            + 0.1 * df["support_tickets_90d"]
            - 0.08 * df["login_days_30d"]
            - 0.02 * df["tenure_months"]
            - 0.15 * df["usage_ratio"]
            - 0.03 * df["nps_score"]
            - 2.5
        )
        probability = 1 / (1 + np.exp(-logit))
        df["churn"] = (rng.random(n_rows) < probability).astype(int)

    return df


def generate_sample_datasets(
    train_path: str | Path = "data/raw/churn_train.csv",
    score_path: str | Path = "data/raw/churn_score.csv",
    feedback_path: str | Path = "data/raw/churn_feedback.csv",
) -> None:
    train_df = _build_dataframe(n_rows=3000, with_target=True, seed=42)
    score_df = _build_dataframe(n_rows=300, with_target=False, seed=2026)
    feedback_df = _build_dataframe(n_rows=250, with_target=True, seed=11)

    for path, dataframe in [
        (train_path, train_df),
        (score_path, score_df),
        (feedback_path, feedback_df),
    ]:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output, index=False)


if __name__ == "__main__":
    generate_sample_datasets()
    print("Arquivos de exemplo gerados em data/raw.")
