from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent import classify_and_route_ticket
from rag import PolicyRAG


PROJECT_DIR = Path(__file__).parent
INPUT_TICKETS = PROJECT_DIR / "input" / "tickets.csv"
POLICY_DOCS_DIR = PROJECT_DIR / "input" / "policy_docs"
OUTPUT_DIR = PROJECT_DIR / "output"


def save_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    tickets = pd.read_csv(INPUT_TICKETS)
    rag = PolicyRAG(POLICY_DOCS_DIR)

    classified_tickets = []
    agent_traces = []
    hallucination_reports = []
    rag_results = []

    for row in tickets.to_dict(orient="records"):
        result = classify_and_route_ticket(
            ticket=row,
            rag=rag,
            policy_docs_dir=POLICY_DOCS_DIR,
        )

        classification = result["classification"]
        trace = result["trace"]
        hallucination = result["hallucination_check"]

        classified_tickets.append(classification.model_dump())
        agent_traces.append(trace.model_dump())
        hallucination_reports.append(hallucination.model_dump())

        rag_results.append(
            {
                "ticket_id": classification.ticket_id,
                "query": row["issue_description"],
                "used_sources": classification.used_sources,
                "policy_context": [
                    {
                        "source_id": doc.source_id,
                        "score": doc.score,
                    }
                    for doc in classification.policy_context
                ],
            }
        )

    save_json(OUTPUT_DIR / "classified_tickets.json", classified_tickets)
    save_json(OUTPUT_DIR / "agent_trace.json", agent_traces)
    save_json(OUTPUT_DIR / "hallucination_report.json", hallucination_reports)
    save_json(OUTPUT_DIR / "rag_results.json", rag_results)

    output_df = pd.DataFrame(classified_tickets)

    stats = {
        "total_tickets": len(output_df),
        "category_counts": output_df["category"].value_counts().to_dict(),
        "priority_counts": output_df["priority"].value_counts().to_dict(),
        "department_counts": output_df["department"].value_counts().to_dict(),
        "escalation_required_count": int(output_df["escalation_required"].sum()),
        "avg_confidence": round(float(output_df["confidence"].mean()), 4),
        "ghost_quote_count": int(sum(item["ghost_quote_count"] for item in hallucination_reports)),
        "fake_source_count": int(sum(item["fake_source_count"] for item in hallucination_reports)),
    }

    save_json(OUTPUT_DIR / "pipeline_summary.json", stats)

    aspect_stats = (
        output_df.groupby(["category", "department"])
        .size()
        .reset_index(name="tickets")
        .sort_values(["category", "department"])
    )
    aspect_stats.to_csv(OUTPUT_DIR / "aspect_stats.csv", index=False)

    final_summary = [
        "# Final pipeline summary",
        "",
        f"Processed tickets: {stats['total_tickets']}",
        f"Average confidence: {stats['avg_confidence']}",
        f"Escalation required: {stats['escalation_required_count']}",
        f"Ghost quote count: {stats['ghost_quote_count']}",
        f"Fake source count: {stats['fake_source_count']}",
        "",
        "## Category counts",
    ]

    for key, value in stats["category_counts"].items():
        final_summary.append(f"- {key}: {value}")

    final_summary.extend(["", "## Department counts"])

    for key, value in stats["department_counts"].items():
        final_summary.append(f"- {key}: {value}")

    (OUTPUT_DIR / "final_summary.md").write_text(
        "\n".join(final_summary),
        encoding="utf-8",
    )

    print("Pipeline finished.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nCreated output files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print("-", path.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
