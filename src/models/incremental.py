from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from src.features.build_features import prepare_features


def update_personalized_model(
    df_labeled: pd.DataFrame,
    model_path: str | Path = "models/churn_model.joblib",
    target_column: str = "churn",
) -> dict[str, Any]:
    payload = joblib.load(model_path)

    if "models" not in payload or "model_roles" not in payload:
        raise RuntimeError(
            "Artefato incompatível para atualização incremental. Treine novamente o modelo."
        )

    personalized_name = payload["model_roles"].get("personalized")
    if personalized_name not in payload["models"]:
        raise RuntimeError("Modelo personalizado não encontrado no artefato.")

    pipeline = payload["models"][personalized_name]
    classifier = pipeline.named_steps["model"]
    if not isinstance(classifier, SGDClassifier):
        raise RuntimeError("O modelo personalizado atual não suporta partial_fit.")

    bundle = prepare_features(
        df_labeled,
        target_column=target_column,
        numeric_features=payload["numeric_features"],
        categorical_features=payload["categorical_features"],
    )
    if bundle.y is None:
        raise ValueError("Dados de atualização precisam conter a coluna alvo 'churn'.")

    transformed_X = pipeline.named_steps["preprocess"].transform(bundle.X)
    classifier.partial_fit(transformed_X, bundle.y.astype(int), classes=np.array([0, 1]))
    payload["incremental_updated_at"] = datetime.now(timezone.utc).isoformat()

    joblib.dump(payload, model_path)
    return {
        "updated_rows": int(len(bundle.X)),
        "personalized_model": personalized_name,
        "model_path": str(Path(model_path).resolve()),
    }
