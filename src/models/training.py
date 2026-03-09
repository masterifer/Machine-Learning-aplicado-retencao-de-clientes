from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import FeatureBundle, prepare_features

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - fallback when dependency is not available.
    XGBClassifier = None


def _build_preprocessor(bundle: FeatureBundle) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, bundle.numeric_features),
            ("cat", categorical_pipeline, bundle.categorical_features),
        ]
    )


def _to_probability(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model = estimator.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        decision = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-decision))

    return estimator.predict(X).astype(float)


def _candidate_models(class_ratio: float) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "sgd_classifier": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-4,
            max_iter=2500,
            random_state=42,
        )
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            scale_pos_weight=class_ratio,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    return models


def _evaluate(y_true: pd.Series, probas: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probas >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probas)),
        "pr_auc": float(average_precision_score(y_true, probas)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def train_churn_model(
    df: pd.DataFrame,
    target_column: str = "churn",
    model_output_path: str | Path = "models/churn_model.joblib",
    metrics_output_path: str | Path = "reports/model_metrics.json",
) -> dict[str, Any]:
    bundle = prepare_features(df, target_column=target_column)
    if bundle.y is None:
        raise ValueError("Dataset sem coluna alvo 'churn'.")
    if bundle.y.nunique() < 2:
        raise ValueError("A coluna 'churn' precisa conter pelo menos duas classes (0 e 1).")

    X_train, X_valid, y_train, y_valid = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=42,
        stratify=bundle.y if bundle.y.nunique() > 1 else None,
    )

    positives = max(1, int((y_train == 1).sum()))
    negatives = max(1, int((y_train == 0).sum()))
    class_ratio = negatives / positives

    candidates = _candidate_models(class_ratio=class_ratio)
    if not candidates:
        raise RuntimeError("Nenhum modelo disponível para treinamento.")

    preprocessor = _build_preprocessor(bundle)
    candidate_metrics: dict[str, dict[str, float]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        probas = _to_probability(pipeline, X_valid)
        metrics = _evaluate(y_valid, probas)
        candidate_metrics[model_name] = metrics
        fitted_pipelines[model_name] = pipeline

    best_model_name = max(
        candidate_metrics,
        key=lambda name: (candidate_metrics[name]["roc_auc"], candidate_metrics[name]["pr_auc"]),
    )
    base_model_name = "xgboost" if "xgboost" in fitted_pipelines else best_model_name
    personalized_model_name = (
        "sgd_classifier" if "sgd_classifier" in fitted_pipelines else best_model_name
    )

    model_payload = {
        "models": fitted_pipelines,
        "best_model_name": best_model_name,
        "model_roles": {
            "base": base_model_name,
            "personalized": personalized_model_name,
        },
        "ensemble_weights": {
            "base": 0.7,
            "personalized": 0.3,
        },
        "target_column": target_column,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, model_path)

    report_payload = {
        "best_model": best_model_name,
        "base_model": base_model_name,
        "personalized_model": personalized_model_name,
        "metrics": candidate_metrics,
        "validation_rows": int(len(X_valid)),
        "train_rows": int(len(X_train)),
    }

    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return report_payload
