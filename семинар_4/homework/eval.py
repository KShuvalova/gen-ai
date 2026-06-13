from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline import RAGPipeline


DATA_DIR = Path("data")
GOLD_PATH = DATA_DIR / "gold.json"
TOP_K = 5


def load_gold() -> list[dict]:
    with GOLD_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_strategy(strategy: str) -> tuple[float, list[dict]]:
    gold_items = load_gold()

    pipeline = RAGPipeline(
        data_dir=DATA_DIR,
        strategy=strategy,
        top_k=TOP_K,
    )

    pipeline.load_documents()
    pipeline.build_chunks()
    pipeline.build_index()

    details: list[dict] = []
    hits = 0

    for item in gold_items:
        retrieved = pipeline.retrieve(item["question"], top_k=TOP_K)
        retrieved_sources = [result["source"] for result in retrieved]
        gold_sources = item["gold_sources"]

        hit = any(source in retrieved_sources for source in gold_sources)

        if hit:
            hits += 1

        details.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "gold_sources": gold_sources,
                "retrieved_sources": retrieved_sources,
                "hit": hit,
                "top_chunks": retrieved,
            }
        )

    hit_rate = hits / len(gold_items)
    return hit_rate, details


def save_details(strategy: str, details: list[dict]) -> None:
    output_path = Path(f"results_{strategy}.json")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)


def save_summary(rows: list[dict]) -> None:
    output_path = Path("eval_results.csv")
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["strategy", "hit_rate@5", "hits", "total"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    strategies = ["fixed", "recursive"]
    summary_rows: list[dict] = []

    for strategy in strategies:
        hit_rate, details = evaluate_strategy(strategy)
        save_details(strategy, details)

        hits = sum(1 for item in details if item["hit"])
        total = len(details)

        summary_rows.append(
            {
                "strategy": strategy,
                "hit_rate@5": round(hit_rate, 4),
                "hits": hits,
                "total": total,
            }
        )

    save_summary(summary_rows)

    print("\nEvaluation results")
    print("------------------")
    for row in summary_rows:
        print(
            f"{row['strategy']}: "
            f"hit-rate@5 = {row['hit_rate@5']} "
            f"({row['hits']}/{row['total']})"
        )

    print("\nSaved files:")
    print("eval_results.csv")
    print("results_fixed.json")
    print("results_recursive.json")


if __name__ == "__main__":
    main()
