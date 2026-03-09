from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allows running `python src/main.py` (Code Runner) and `python -m src.main`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sistema de previsão de churn.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Treina o modelo de churn.")
    train_parser.add_argument("--train-path", default="data/raw/churn_train.csv")
    train_parser.add_argument("--model-output", default="models/churn_model.joblib")
    train_parser.add_argument("--metrics-output", default="reports/model_metrics.json")

    score_parser = subparsers.add_parser("score", help="Gera score de churn.")
    score_parser.add_argument("--input-path", default="data/raw/churn_score.csv")
    score_parser.add_argument("--model-path", default="models/churn_model.joblib")
    score_parser.add_argument("--output-path", default="data/processed/churn_scores.csv")

    update_parser = subparsers.add_parser(
        "update", help="Atualiza o modelo personalizado com dados rotulados novos."
    )
    update_parser.add_argument("--labeled-path", default="data/raw/churn_feedback.csv")
    update_parser.add_argument("--model-path", default="models/churn_model.joblib")

    return parser


def run_train(args: argparse.Namespace) -> None:
    from src.data.dataset import load_dataset
    from src.models.training import train_churn_model

    df = load_dataset(args.train_path, require_target=True)
    report = train_churn_model(
        df=df,
        target_column="churn",
        model_output_path=args.model_output,
        metrics_output_path=args.metrics_output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_score(args: argparse.Namespace) -> None:
    from src.data.dataset import load_dataset
    from src.models.inference import score_customers

    df = load_dataset(args.input_path, require_target=False)
    output = score_customers(df=df, model_path=args.model_path, output_path=args.output_path)
    print(output.head(10).to_string(index=False))
    print(f"\nArquivo gerado: {Path(args.output_path).resolve()}")


def run_update(args: argparse.Namespace) -> None:
    from src.data.dataset import load_dataset
    from src.models.incremental import update_personalized_model

    df = load_dataset(args.labeled_path, require_target=True)
    report = update_personalized_model(
        df_labeled=df,
        model_path=args.model_path,
        target_column="churn",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
        return
    if args.command == "score":
        run_score(args)
        return
    if args.command == "update":
        run_update(args)
        return

    parser.error(f"Comando inválido: {args.command}")


if __name__ == "__main__":
    main()
