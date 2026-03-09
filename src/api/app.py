from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
import socket
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allows running `python src/api/app.py` in IDE runners.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import score_customers


class PredictRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(
        ..., description="Lista de clientes com as features para inferência."
    )


@lru_cache(maxsize=1)
def _load_model_payload() -> dict[str, Any]:
    model_path = os.getenv("CHURN_MODEL_PATH", "models/churn_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Execute o treino antes de subir a API."
        )
    return joblib.load(model_path)


app = FastAPI(
    title="Churn Prediction API",
    description="API para geração de score de risco de churn e ação recomendada.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        payload = _load_model_payload()
        return {
            "status": "ok",
            "model": payload.get("best_model_name", payload.get("model_name", "unknown")),
        }
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "details": str(exc)}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if not request.customers:
        raise HTTPException(status_code=400, detail="A lista de clientes está vazia.")

    _load_model_payload()  # Validate model availability early.
    df = pd.DataFrame(request.customers)
    scores = score_customers(df=df, output_path=None)
    return {
        "count": int(len(scores)),
        "results": scores.to_dict(orient="records"),
    }


if __name__ == "__main__":
    import uvicorn

    def _is_port_available(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.connect_ex((host, port)) != 0

    def _pick_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
        for port in range(preferred_port, preferred_port + max_tries):
            if _is_port_available(host, port):
                return port
        raise RuntimeError(
            f"Nenhuma porta disponível entre {preferred_port} e {preferred_port + max_tries - 1}."
        )

    host = os.getenv("HOST", "127.0.0.1")
    preferred_port = int(os.getenv("PORT", "8000"))
    selected_port = _pick_port(host, preferred_port)

    if selected_port != preferred_port:
        print(
            f"Porta {preferred_port} ocupada. Subindo API na porta {selected_port}."
        )

    uvicorn.run("src.api.app:app", host=host, port=selected_port, reload=False)
