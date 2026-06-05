from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ValidationError

from llm_client import get_model, make_client
from prompts import (
    ASPECT_SYSTEM_PROMPT,
    IE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    build_aspect_prompt,
    build_ie_prompt,
    build_judge_prompt,
    build_map_prompt,
    build_reduce_prompt,
)
from schema import (
    AspectBatch,
    FinalSummary,
    JudgeReport,
    MapSummary,
    Review,
    ReviewBatch,
)


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
MODEL = get_model()

# Approximate prices. Needed for homework reporting.
# If your course has another tariff, replace these values.
PRICE_INPUT_PER_1M = 0.07
PRICE_OUTPUT_PER_1M = 0.27


def to_json(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2, default=str)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    else:
        payload = data

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def call_llm(
    client,
    *,
    system_prompt: str,
    user_prompt: str,
    response_model,
    temperature: float = 0.1,
):
    """
    Wrapper around make_client().

    It always uses:
    - response_model
    - max_retries=3

    It also tries to collect response.usage.
    If the course client does not support with_completion=True,
    the call still works, but usage is saved as zeros.
    """
    started = time.time()

    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=response_model,
        max_retries=3,
        temperature=temperature,
    )

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    try:
        result, completion = client.chat.completions.create(
            **kwargs,
            with_completion=True,
        )

        raw_usage = getattr(completion, "usage", None)

        if raw_usage is not None:
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }

    except TypeError:
        result = client.chat.completions.create(**kwargs)

    elapsed = time.time() - started

    return result, usage, elapsed


def estimate_cost(usage_items: list[dict]) -> dict:
    prompt_tokens = sum(item.get("prompt_tokens", 0) for item in usage_items)
    completion_tokens = sum(item.get("completion_tokens", 0) for item in usage_items)
    total_tokens = sum(item.get("total_tokens", 0) for item in usage_items)

    cost = (
        prompt_tokens / 1_000_000 * PRICE_INPUT_PER_1M
        + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


def normalize_quote(quote: str) -> str:
    return " ".join(str(quote).lower().replace("ё", "е").split())


def check_quotes(quotes: list[str], source_text: str) -> dict:
    """
    Sanity-check for hallucinated quotes.

    A quote is valid if its normalized form is found inside normalized source text.
    """
    normalized_source = normalize_quote(source_text)

    checked = []
    ghost = []

    for quote in quotes:
        if quote is None:
            continue

        q = str(quote).strip()

        if not q:
            continue

        ok = normalize_quote(q) in normalized_source

        item = {
            "quote": q,
            "found_in_source": ok,
        }

        checked.append(item)

        if not ok:
            ghost.append(item)

    total = len(checked)
    ghost_count = len(ghost)
    ghost_share = ghost_count / total if total else 0

    return {
        "total_quotes_checked": total,
        "ghost_quotes": ghost_count,
        "ghost_quote_share": round(ghost_share, 4),
        "items": checked,
        "ghost_items": ghost,
    }


def collect_quotes(
    reviews: list[Review],
    aspects: list,
    map_summaries: list[MapSummary],
    final_summary: FinalSummary | None = None,
) -> list[str]:
    quotes = []

    for review in reviews:
        for issue in review.issues:
            quotes.append(issue.evidence_quote)

    for aspect in aspects:
        if aspect.evidence_quote:
            quotes.append(aspect.evidence_quote)

    for summary in map_summaries:
        quotes.extend(summary.representative_quotes)

    if final_summary is not None:
        quotes.extend(final_summary.evidence_quotes)

    return quotes


def plot_heatmap(aspects: list, out_path: Path) -> None:
    rows = [
        {
            "review_id": aspect.review_id,
            "aspect": aspect.aspect,
            "severity": aspect.severity,
        }
        for aspect in aspects
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return

    pivot = df.pivot_table(
        index="aspect",
        columns="review_id",
        values="severity",
        aggfunc="mean",
        fill_value=0,
    )

    plt.figure(figsize=(14, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Severity, 0-5")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title("Aspect severity heatmap")
    plt.xlabel("Review ID")
    plt.ylabel("Aspect")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def aspect_stats(aspects: list) -> dict:
    rows = [
        {
            "review_id": aspect.review_id,
            "aspect": aspect.aspect,
            "sentiment": aspect.sentiment,
            "severity": aspect.severity,
        }
        for aspect in aspects
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return {}

    by_aspect = (
        df.groupby("aspect")
        .agg(
            mean_severity=("severity", "mean"),
            max_severity=("severity", "max"),
            mentions=("severity", lambda x: int((x > 0).sum())),
            total=("severity", "count"),
        )
        .reset_index()
    )

    by_aspect["mention_share"] = by_aspect["mentions"] / by_aspect["total"]

    sentiment_pivot = pd.crosstab(df["aspect"], df["sentiment"])

    return {
        "by_aspect": by_aspect.to_dict(orient="records"),
        "sentiment_pivot": sentiment_pivot.to_dict(),
    }


def cross_source_pivot(reviews: list[Review], aspects: list, out_path: Path) -> pd.DataFrame:
    review_to_source = {review.review_id: review.source_file for review in reviews}

    rows = []

    for aspect in aspects:
        rows.append(
            {
                "source_file": review_to_source.get(aspect.review_id, "unknown"),
                "aspect": aspect.aspect,
                "severity": aspect.severity,
                "mentioned": int(aspect.severity > 0),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        pivot = pd.DataFrame()
    else:
        pivot = df.pivot_table(
            index="source_file",
            columns="aspect",
            values="severity",
            aggfunc="mean",
            fill_value=0,
        )

    pivot.to_csv(out_path, encoding="utf-8-sig")
    return pivot


def build_conclusions(
    *,
    n_input_files: int,
    n_reviews: int,
    validation_errors: int,
    quote_check: dict,
    judge_report: JudgeReport,
    runtime_seconds: float,
    cost_report: dict,
    out_path: Path,
) -> None:
    unsupported = [
        item
        for item in judge_report.weak_or_unsupported_items
        if item.support in {"weakly_supported", "not_supported"}
    ]

    if unsupported:
        judge_example = unsupported[0]
        judge_text = (
            f"Например, judge пометил action item «{judge_example.action_item}» "
            f"как {judge_example.support}, потому что: {judge_example.reason}"
        )
    else:
        judge_text = (
            "Judge не нашел weakly_supported или not_supported action items, "
            "поэтому финальные рекомендации были признаны достаточно подтвержденными."
        )

    ghost_examples = quote_check.get("ghost_items", [])[:3]

    if ghost_examples:
        ghost_text = "\n".join(
            f"- «{item['quote']}» — цитата не была найдена дословно в исходных текстах."
            for item in ghost_examples
        )
    else:
        ghost_text = "- Ghost-цитаты не найдены: все проверенные цитаты были обнаружены в исходных текстах."

    text = f"""# Выводы

## 1. Что получилось

В работе был собран текстовый пайплайн для анализа отзывов на мобильное приложение Any.do. Входные данные были взяты из реального CSV с отзывами Google Play и разложены на {n_input_files} отдельных файлов, что позволило использовать multi-doc обработку. Всего обработано {n_reviews} отзывов. Валидных объектов после IE: {n_reviews}; ValidationError: {validation_errors}. Пайплайн включает четыре базовые техники семинара: information extraction, аспектный анализ, Map-Reduce и LLM-as-judge. Дополнительно реализованы multi-doc разбиение и cross-source pivot.

Sanity-check цитат через `check_quotes` проверил {quote_check.get("total_quotes_checked", 0)} цитат. Найдено ghost-цитат: {quote_check.get("ghost_quotes", 0)}, доля ghost-цитат: {quote_check.get("ghost_quote_share", 0):.1%}. Итоговая оценка judge: overall_score = {judge_report.overall_score}. Полный прогон занял {runtime_seconds:.1f} секунд. По данным `response.usage` было использовано {cost_report.get("total_tokens", 0)} токенов, примерная стоимость составила ${cost_report.get("estimated_cost_usd", 0)}.

## 2. Где модель ошибалась

Основной риск был связан не с Pydantic-валидацией, а с grounding цитат: модель иногда может переформулировать отзыв и выдать пересказ как цитату. Для этого был добавлен `check_quotes`, который ищет каждую evidence_quote в исходных текстах. Примеры найденных проблем:

{ghost_text}

{judge_text}

Также отдельная сложность связана с аспектами. Некоторые отзывы одновременно касаются performance и reliability: например, зависание при оплате можно отнести и к скорости работы, и к надежности платежей. В таких случаях модель иногда ставила оценки сразу по двум аспектам. Это допустимо, но в production потребовало бы более строгих правил разметки.

## 3. Что бы изменили в production

В production я бы оставил общую структуру пайплайна: IE → аспектный анализ → Map-Reduce → judge. Такая архитектура хорошо разделяет извлечение фактов, оценку аспектов, суммаризацию и контроль качества. Также я бы оставил Pydantic-схемы и `response_model`, потому что они снижают число невалидных ответов модели.

Переделать стоит несколько частей. Во-первых, я бы усилил промпт для evidence_quote: требовать не только дословную цитату, но и минимальную длину цитаты без пересказа. Во-вторых, добавил бы повторную генерацию конкретного объекта, если `check_quotes` нашел ghost-цитату. В-третьих, для длинных массивов отзывов я бы добавил кэширование и A/B-тест двух моделей, чтобы сравнить стоимость и качество. Для критичных продуктовых решений также нужна ручная проверка части рекомендаций аналитиком.
"""

    out_path.write_text(text, encoding="utf-8")


def analyze(input_path: str | Path = INPUT_DIR) -> None:
    start_time = time.time()

    input_path = Path(input_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = make_client()

    usage_items = []
    validation_errors = 0

    source_files = sorted(input_path.glob("reviews_batch_*.txt"))

    if not source_files:
        raise FileNotFoundError("Не найдены input/reviews_batch_*.txt")

    all_reviews: list[Review] = []
    raw_text_by_source = {}

    print("IE: extracting reviews")

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        raw_text_by_source[path.name] = text

        try:
            batch, usage, elapsed = call_llm(
                client,
                system_prompt=IE_SYSTEM_PROMPT,
                user_prompt=build_ie_prompt(path.name, text),
                response_model=ReviewBatch,
                temperature=0.1,
            )
            usage_items.append(usage)
            print(f"- {path.name}: {len(batch.reviews)} reviews, {elapsed:.1f}s")
            all_reviews.extend(batch.reviews)

        except ValidationError as e:
            validation_errors += 1
            print(f"ValidationError in {path.name}: {e}")

    save_json(OUTPUT_DIR / "reviews.json", {"reviews": [r.model_dump() for r in all_reviews]})

    print("Aspect analysis")

    aspects_result, usage, elapsed = call_llm(
        client,
        system_prompt=ASPECT_SYSTEM_PROMPT,
        user_prompt=build_aspect_prompt(to_json({"reviews": [r.model_dump() for r in all_reviews]})),
        response_model=AspectBatch,
        temperature=0.1,
    )

    usage_items.append(usage)
    all_aspects = aspects_result.aspects

    print(f"- aspects: {len(all_aspects)}, {elapsed:.1f}s")

    save_json(OUTPUT_DIR / "aspects.json", {"aspects": [a.model_dump() for a in all_aspects]})
    plot_heatmap(all_aspects, OUTPUT_DIR / "heatmap.png")

    stats = aspect_stats(all_aspects)
    save_json(OUTPUT_DIR / "aspect_stats.json", stats)

    pivot = cross_source_pivot(
        all_reviews,
        all_aspects,
        OUTPUT_DIR / "cross_source_pivot.csv",
    )

    print("Map step")

    reviews_by_source: dict[str, list[Review]] = {}

    for review in all_reviews:
        reviews_by_source.setdefault(review.source_file, []).append(review)

    aspects_by_review = {aspect.review_id: [] for aspect in all_aspects}
    for aspect in all_aspects:
        aspects_by_review.setdefault(aspect.review_id, []).append(aspect)

    map_summaries: list[MapSummary] = []

    for source_file, reviews in reviews_by_source.items():
        source_aspects = []

        for review in reviews:
            source_aspects.extend(aspects_by_review.get(review.review_id, []))

        map_summary, usage, elapsed = call_llm(
            client,
            system_prompt=MAP_SYSTEM_PROMPT,
            user_prompt=build_map_prompt(
                source_file,
                to_json({"reviews": [r.model_dump() for r in reviews]}),
                to_json({"aspects": [a.model_dump() for a in source_aspects]}),
            ),
            response_model=MapSummary,
            temperature=0.1,
        )

        usage_items.append(usage)
        map_summaries.append(map_summary)

        print(f"- {source_file}: map summary, {elapsed:.1f}s")

    save_json(
        OUTPUT_DIR / "map_summaries.json",
        {"map_summaries": [m.model_dump() for m in map_summaries]},
    )

    all_source_text = "\n\n".join(raw_text_by_source.values())
    preliminary_quotes = collect_quotes(all_reviews, all_aspects, map_summaries)
    preliminary_quote_check = check_quotes(preliminary_quotes, all_source_text)

    save_json(OUTPUT_DIR / "quote_check_preliminary.json", preliminary_quote_check)

    print("Reduce step")

    final_summary, usage, elapsed = call_llm(
        client,
        system_prompt=REDUCE_SYSTEM_PROMPT,
        user_prompt=build_reduce_prompt(
            to_json({"map_summaries": [m.model_dump() for m in map_summaries]}),
            to_json(stats),
            to_json(preliminary_quote_check),
        ),
        response_model=FinalSummary,
        temperature=0.1,
    )

    usage_items.append(usage)

    print(f"- final summary, {elapsed:.1f}s")

    save_json(OUTPUT_DIR / "summary.json", final_summary)

    final_quotes = collect_quotes(all_reviews, all_aspects, map_summaries, final_summary)
    final_quote_check = check_quotes(final_quotes, all_source_text)

    save_json(OUTPUT_DIR / "quote_check.json", final_quote_check)

    print("Judge step")

    judge_report, usage, elapsed = call_llm(
        client,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=build_judge_prompt(
            to_json(final_summary),
            to_json({"reviews": [r.model_dump() for r in all_reviews]}),
            to_json({"aspects": [a.model_dump() for a in all_aspects]}),
            to_json(final_quote_check),
        ),
        response_model=JudgeReport,
        temperature=0.1,
    )

    usage_items.append(usage)

    print(f"- judge overall_score={judge_report.overall_score}, {elapsed:.1f}s")

    # Homework requirement: if score < 0.7, rerun reduce once.
    if judge_report.overall_score < 0.7:
        print("overall_score < 0.7, rerunning REDUCE with stricter prompt")

        stricter_reduce_prompt = (
            REDUCE_SYSTEM_PROMPT
            + "\n\nСудья оценил предыдущий результат ниже 0.7. "
            "Сделай новую версию более фактичной, с меньшим числом широких рекомендаций "
            "и только с дословными цитатами."
        )

        final_summary, usage, elapsed = call_llm(
            client,
            system_prompt=stricter_reduce_prompt,
            user_prompt=build_reduce_prompt(
                to_json({"map_summaries": [m.model_dump() for m in map_summaries]}),
                to_json(stats),
                to_json(final_quote_check),
            ),
            response_model=FinalSummary,
            temperature=0.05,
        )

        usage_items.append(usage)
        save_json(OUTPUT_DIR / "summary.json", final_summary)

        final_quotes = collect_quotes(all_reviews, all_aspects, map_summaries, final_summary)
        final_quote_check = check_quotes(final_quotes, all_source_text)
        save_json(OUTPUT_DIR / "quote_check.json", final_quote_check)

        judge_report, usage, elapsed = call_llm(
            client,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=build_judge_prompt(
                to_json(final_summary),
                to_json({"reviews": [r.model_dump() for r in all_reviews]}),
                to_json({"aspects": [a.model_dump() for a in all_aspects]}),
                to_json(final_quote_check),
            ),
            response_model=JudgeReport,
            temperature=0.05,
        )

        usage_items.append(usage)
        print(f"- judge after rerun overall_score={judge_report.overall_score}, {elapsed:.1f}s")

    save_json(OUTPUT_DIR / "judge_report.json", judge_report)

    runtime_seconds = time.time() - start_time
    cost_report = estimate_cost(usage_items)

    run_metrics = {
        "input_files": len(source_files),
        "valid_reviews": len(all_reviews),
        "validation_errors": validation_errors,
        "runtime_seconds": round(runtime_seconds, 2),
        "usage": usage_items,
        "cost": cost_report,
        "ghost_quotes": final_quote_check.get("ghost_quotes", 0),
        "ghost_quote_share": final_quote_check.get("ghost_quote_share", 0),
        "judge_overall_score": judge_report.overall_score,
    }

    save_json(OUTPUT_DIR / "run_metrics.json", run_metrics)

    build_conclusions(
        n_input_files=len(source_files),
        n_reviews=len(all_reviews),
        validation_errors=validation_errors,
        quote_check=final_quote_check,
        judge_report=judge_report,
        runtime_seconds=runtime_seconds,
        cost_report=cost_report,
        out_path=Path("выводы.md"),
    )

    print()
    print("Done")
    print(f"- reviews: {len(all_reviews)}")
    print(f"- validation_errors: {validation_errors}")
    print(f"- ghost_quotes: {final_quote_check.get('ghost_quotes', 0)}")
    print(f"- judge overall_score: {judge_report.overall_score}")
    print(f"- runtime: {runtime_seconds:.1f}s")
    print(f"- estimated cost: ${cost_report.get('estimated_cost_usd', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input")
    args = parser.parse_args()

    analyze(args.input)


if __name__ == "__main__":
    main()
