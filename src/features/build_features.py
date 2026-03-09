from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_NUMERIC_FEATURES = [
    "tenure_months",
    "recency_days",
    "purchase_frequency_90d",
    "avg_ticket",
    "support_tickets_90d",
    "payment_delay_days",
    "failed_payments_90d",
    "login_days_30d",
    "engagement_30d",
    "usage_ratio",
    "nps_score",
    "satisfaction_score",
    "plan_value",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "plan_type",
    "contract_type",
    "payment_method",
    "region",
]


@dataclass
class FeatureBundle:
    X: pd.DataFrame
    y: Optional[pd.Series]
    numeric_features: list[str]
    categorical_features: list[str]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _to_datetime_if_exists(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    _to_datetime_if_exists(frame, ["snapshot_date", "last_purchase_date", "signup_date"])

    if "snapshot_date" in frame.columns and "last_purchase_date" in frame.columns and "recency_days" not in frame.columns:
        frame["recency_days"] = (frame["snapshot_date"] - frame["last_purchase_date"]).dt.days

    if "snapshot_date" in frame.columns and "signup_date" in frame.columns and "tenure_months" not in frame.columns:
        tenure_days = (frame["snapshot_date"] - frame["signup_date"]).dt.days
        frame["tenure_months"] = tenure_days / 30.44

    if "total_revenue_90d" in frame.columns and "purchase_frequency_90d" in frame.columns and "avg_ticket" not in frame.columns:
        frame["avg_ticket"] = _safe_divide(frame["total_revenue_90d"], frame["purchase_frequency_90d"])

    if "login_days_30d" in frame.columns and "engagement_30d" not in frame.columns:
        frame["engagement_30d"] = _safe_divide(frame["login_days_30d"], pd.Series(30, index=frame.index))

    if "support_tickets_90d" in frame.columns and "purchase_frequency_90d" in frame.columns:
        frame["support_to_purchase_ratio"] = _safe_divide(
            frame["support_tickets_90d"], frame["purchase_frequency_90d"] + 1
        )

    if "failed_payments_90d" in frame.columns and "payment_delay_days" in frame.columns:
        frame["payment_risk_score"] = frame["failed_payments_90d"].fillna(0) + (
            frame["payment_delay_days"].fillna(0) / 10
        )

    if "nps_score" in frame.columns and "satisfaction_score" in frame.columns:
        frame["experience_score"] = (frame["nps_score"] + frame["satisfaction_score"]) / 2

    return frame


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "churn",
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    include_engineered_features: bool = True,
) -> FeatureBundle:
    frame = _derive_features(df) if include_engineered_features else df.copy()
    numeric = list(numeric_features or DEFAULT_NUMERIC_FEATURES)
    categorical = list(categorical_features or DEFAULT_CATEGORICAL_FEATURES)

    # Add derived features if available to improve predictive signal.
    for extra_numeric in ["support_to_purchase_ratio", "payment_risk_score", "experience_score"]:
        if extra_numeric in frame.columns and extra_numeric not in numeric:
            numeric.append(extra_numeric)

    selected_columns = []
    for col in numeric + categorical:
        if col in frame.columns:
            selected_columns.append(col)
        else:
            frame[col] = np.nan
            selected_columns.append(col)

    X = frame[selected_columns].copy()
    y = frame[target_column].copy() if target_column in frame.columns else None

    return FeatureBundle(X=X, y=y, numeric_features=numeric, categorical_features=categorical)
