from __future__ import annotations


IE_SYSTEM_PROMPT = """
Ты выполняешь information extraction для отзывов о мобильном приложении.

Задача:
- прочитать текстовый файл с отзывами;
- извлечь каждый отзыв как отдельный объект Review;
- сохранить исходный текст отзыва в поле review_text;
- извлечь проблемы пользователя в поле issues;
- evidence_quote должен быть дословной цитатой из review_text;
- не придумывай факты, которых нет в тексте;
- если дата не указана, поставь review_date = null;
- если источник не указан, поставь source = "Other";
- app_name должен быть одинаковым для всех отзывов: "Any.do";
- source_file заполняй именем файла, которое передано в промпте.

Отвечай строго по Pydantic-схеме ReviewBatch.
""".strip()


ASPECT_SYSTEM_PROMPT = """
Ты выполняешь аспектный анализ отзывов о мобильном приложении.

Фиксированные аспекты:
1. performance — скорость, лаги, зависания, загрузка;
2. design — интерфейс, удобство, навигация, визуальная часть;
3. support — поддержка, ответы операторов, обработка обращений;
4. price — цены, комиссии, платные функции, подписка;
5. ads — реклама, пуши, навязчивые уведомления;
6. reliability — стабильность, ошибки, вход, платежи, заказы, доверие.

Для каждого review_id оцени каждый аспект.
Если аспект не упомянут:
- sentiment = "not_mentioned";
- severity = 0;
- evidence_quote = null.

Если аспект упомянут:
- sentiment: positive, neutral, negative или mixed;
- severity от 1 до 5, где 1 — слабая проблема или слабый сигнал, 5 — критичная проблема;
- evidence_quote должна быть дословной цитатой из review_text.

Не придумывай цитаты.
Отвечай строго по Pydantic-схеме AspectBatch.
""".strip()


MAP_SYSTEM_PROMPT = """
Ты делаешь MAP-этап Map-Reduce анализа отзывов.

Тебе будет передан один источник с отзывами и аспектными оценками.
Нужно кратко свернуть только этот источник.

Правила:
- используй только переданные отзывы и аспектные оценки;
- не добавляй внешние знания;
- representative_quotes должны быть дословными цитатами из отзывов;
- key_findings должны быть конкретными, а не общими;
- frequent_issues должны отражать повторяющиеся проблемы.

Отвечай строго по Pydantic-схеме MapSummary.
""".strip()


REDUCE_SYSTEM_PROMPT = """
Ты делаешь REDUCE-этап Map-Reduce анализа отзывов о мобильном приложении.

Тебе будут переданы результаты MAP-этапа по нескольким файлам, статистика аспектов и проверка цитат.
Нужно сделать общий итоговый summary.

Правила:
- не придумывай новые отзывы;
- evidence_quotes должны быть только из переданных representative_quotes;
- action_items должны быть практическими и связанными с найденными проблемами;
- если проблема упоминается часто, явно отметь это;
- если проблема редкая, не делай ее главным выводом;
- summary должно быть полезным для продуктовой команды.

Отвечай строго по Pydantic-схеме FinalSummary.
""".strip()


JUDGE_SYSTEM_PROMPT = """
Ты LLM-as-judge. Твоя задача — оценить качество итогового summary по отзывам.

Оцени:
1. factuality_score — насколько summary соответствует исходным отзывам;
2. usefulness_score — насколько выводы полезны для продуктовой команды;
3. quote_grounding_score — насколько цитаты действительно подтверждают выводы;
4. overall_score — общая оценка от 0 до 1.

Также оцени каждый action_item:
- supported — хорошо подтвержден отзывами;
- weakly_supported — частично подтвержден, но сформулирован слишком широко;
- not_supported — не подтвержден исходными данными.

Если overall_score < 0.7, verdict должен быть "revise".
Если overall_score >= 0.7, verdict может быть "pass".

Отвечай строго по Pydantic-схеме JudgeReport.
""".strip()


def build_ie_prompt(source_file: str, text: str) -> str:
    return f"""
Имя файла: {source_file}

Текст отзывов:
{text}

Извлеки все отзывы из этого файла.
""".strip()


def build_aspect_prompt(reviews_json: str) -> str:
    return f"""
Ниже передан JSON со списком отзывов.

Отзывы:
{reviews_json}

Сделай аспектный анализ для каждого review_id и каждого из 6 фиксированных аспектов.
""".strip()


def build_map_prompt(source_file: str, reviews_json: str, aspects_json: str) -> str:
    return f"""
Источник: {source_file}

Отзывы этого источника:
{reviews_json}

Аспектные оценки этого источника:
{aspects_json}

Сделай краткую MAP-сводку по этому источнику.
""".strip()


def build_reduce_prompt(
    map_summaries_json: str,
    aspect_stats_json: str,
    quote_check_json: str,
) -> str:
    return f"""
MAP-сводки по источникам:
{map_summaries_json}

Статистика аспектов:
{aspect_stats_json}

Результат проверки цитат check_quotes:
{quote_check_json}

Сделай общий итоговый summary по всем отзывам.
""".strip()


def build_judge_prompt(
    final_summary_json: str,
    all_reviews_json: str,
    all_aspects_json: str,
    quote_check_json: str,
) -> str:
    return f"""
Итоговый summary:
{final_summary_json}

Исходные отзывы:
{all_reviews_json}

Аспектные оценки:
{all_aspects_json}

Результат проверки цитат:
{quote_check_json}

Оцени итоговый summary как judge.
""".strip()
