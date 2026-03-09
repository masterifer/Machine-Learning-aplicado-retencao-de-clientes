from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import prepare_features


def _predict_with_estimator(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    model = estimator.named_steps["model"]

    if hasattr(model, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return estimator.predict(X).astype(float)


def _to_probability(model_payload: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    if "model" in model_payload:
        return _predict_with_estimator(model_payload["model"], X)

    models = model_payload["models"]
    roles = model_payload.get("model_roles", {})
    weights = model_payload.get("ensemble_weights", {})
    base_name = roles.get("base")
    personalized_name = roles.get("personalized")

    base_proba = _predict_with_estimator(models[base_name], X) if base_name in models else None
    personalized_proba = (
        _predict_with_estimator(models[personalized_name], X) if personalized_name in models else None
    )

    if base_proba is not None and personalized_proba is not None:
        return (
            weights.get("base", 0.7) * base_proba
            + weights.get("personalized", 0.3) * personalized_proba
        )
    if base_proba is not None:
        return base_proba
    if personalized_proba is not None:
        return personalized_proba

    # Fallback in case no role metadata exists.
    first_model = next(iter(models.values()))
    return _predict_with_estimator(first_model, X)


def _risk_level(score: float) -> str:
    if score >= 0.7:
        return "alto"
    if score >= 0.4:
        return "medio"
    return "baixo"


def _retention_action(row: pd.Series) -> str:
    if row["risk_level"] == "alto":
        return "Contato humano imediato + oferta de retenção personalizada."
    if row["risk_level"] == "medio":
        return "Campanha direcionada com benefício temporário e monitoramento semanal."
    return "Fluxo de engajamento automático e programa de fidelidade."


def score_customers(
    df: pd.DataFrame,
    model_path: str | Path = "models/churn_model.joblib",
    output_path: str | Path | None = "data/processed/churn_scores.csv",
) -> pd.DataFrame:
    payload = joblib.load(model_path)
    bundle = prepare_features(
        df,
        target_column=payload["target_column"],
        numeric_features=payload["numeric_features"],
        categorical_features=payload["categorical_features"],
    )
    probas = _to_probability(payload, bundle.X)

    output = pd.DataFrame(
        {
            "customer_id": df["customer_id"] if "customer_id" in df.columns else pd.RangeIndex(1, len(df) + 1),
            "churn_risk_score": np.round(probas, 4),
        }
    )
    output["risk_level"] = output["churn_risk_score"].apply(_risk_level)
    output["retention_action"] = output.apply(_retention_action, axis=1)
    output = output.sort_values("churn_risk_score", ascending=False).reset_index(drop=True)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(out_path, index=False)
    return output
