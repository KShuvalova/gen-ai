# AI-конвейер для классификации и маршрутизации тикетов поддержки

## 1. Описание проекта

Проект реализует рабочий конвейер для классификации и маршрутизации тикетов клиентской поддержки. Система принимает текст обращения, определяет категорию проблемы, приоритет, ответственный отдел, необходимость эскалации, подбирает релевантные policy-документы и проверяет, что evidence quote действительно взята из исходного тикета.

Проект выполнен в рамках трека B финального проекта по курсу «Практическое применение генеративного ИИ».

## 2. Данные

Источник данных: Customer Support Tickets Dataset 200k Records с Kaggle.

В репозиторий включена только обезличенная и уменьшенная подвыборка:

- `input/tickets.csv` - 100 тикетов;
- `input/eval_cases.json` - 20 размеченных eval-кейсов;
- `input/challenge_cases.json` - 12 сложных кейсов;
- `input/policy_docs/` - база правил маршрутизации для RAG.

Raw-файл на 200000 строк не добавляется в GitHub и исключён через `.gitignore`.

Из исходного датасета были исключены персональные поля:

- `customer_name`;
- `customer_email`.

## 3. Структура проекта

support_ticket_routing/
├── README.md
├── requirements.txt
├── .env.example
├── run_all.py
├── prepare_data.py
├── schema.py
├── rag.py
├── tools.py
├── agent.py
├── pipeline.py
├── eval.py
├── challenge_eval.py
├── llm_client.py
├── llm_judge.py
├── input/
│   ├── tickets.csv
│   ├── eval_cases.json
│   ├── challenge_cases.json
│   ├── dataset_profile.json
│   ├── raw/
│   └── policy_docs/
│       ├── account_access.md
│       ├── billing_refunds.md
│       ├── escalation_policy.md
│       ├── priority_rules.md
│       ├── routing_rules.md
│       ├── subscription_policy.md
│       └── technical_bugs.md
└── output/
    ├── classified_tickets.json
    ├── rag_results.json
    ├── agent_trace.json
    ├── hallucination_report.json
    ├── pipeline_summary.json
    ├── aspect_stats.csv
    ├── eval_results.csv
    ├── eval_detailed.csv
    ├── judge_report.json
    ├── llm_judge_report.json
    ├── llm_judge_results.csv
    ├── challenge_eval.csv
    └── final_summary.md

## 4. Использованные техники курса

В проекте используются следующие техники:

1. Structured output через Pydantic-схемы.
2. `field_validator` для проверки входного текста, источников, цитат и структуры результата.
3. Information extraction из тикета.
4. Classification: категория, приоритет, отдел.
5. RAG по policy-документам.
6. Agent tools: классификация, выбор отдела, проверка эскалации, retrieval, hallucination check.
7. LLM-as-judge через DeepSeek-compatible API.
8. Проверка галлюцинаций: ghost quotes и fake sources.
9. Eval на 20 тестовых кейсах и отдельный challenge set.

## 5. Установка

Команда установки зависимостей:

python -m pip install -r requirements.txt

## 6. Настройка LLM-as-judge

Создать файл `.env` по примеру:

cp .env.example .env

Пример `.env`:

LLM_BASE_URL=https://api.deepseek.com
LLM_AUTH_TOKEN=your_token_here
LLM_MODEL=deepseek-v4-flash

Если токен не указан, основной pipeline и rule-based eval всё равно работают, но `llm_judge.py` будет пропущен.

## 7. Запуск

Полный запуск:

python run_all.py

Отдельные этапы:

python prepare_data.py
python pipeline.py
python eval.py
python challenge_eval.py
python llm_judge.py

## 8. Основные результаты

Pipeline обработал 100 тикетов.

Основной eval на 20 кейсах:

- category_accuracy = 1.0
- priority_accuracy = 1.0
- department_accuracy = 1.0
- escalation_accuracy = 1.0
- quote_valid_rate = 1.0
- rag_hit_rate = 1.0
- agent_path_valid_rate = 1.0
- judge_pass_rate = 1.0
- ghost_quote_count = 0
- fake_source_count = 0

LLM-as-judge:

- model = deepseek-v4-flash
- total = 20
- llm_judge_pass_rate = 1.0
- category_ok_rate = 1.0
- priority_ok_rate = 1.0
- department_ok_rate = 1.0
- evidence_ok_rate = 1.0
- source_ok_rate = 1.0

Challenge set на 12 сложных кейсах:

- category_accuracy = 0.75
- department_accuracy = 0.75
- quote_valid_rate = 1.0
- pass_rate = 0.75
- failed_cases = 3
- ghost_quote_count = 0
- fake_source_count = 0

## 9. Ограничения

Основной eval показывает идеальные метрики, потому что тестовые кейсы были размечены в той же логике, что и rule-based маршрутизация. Поэтому дополнительно был создан challenge set с неоднозначными обращениями. На нём система ошиблась в 3 из 12 кейсов, что показывает реальные ограничения keyword-based классификации.

Основные ограничения:

- rule-based классификация чувствительна к формулировкам;
- неоднозначные тикеты с несколькими проблемами могут быть направлены не в тот отдел;
- RAG использует TF-IDF, а не embedding model;
- policy-документы являются учебной моделью правил, а не реальным регламентом компании.

## 10. Вывод

Проект показывает воспроизводимый AI-конвейер для классификации и маршрутизации тикетов поддержки. Система объединяет structured output, RAG, agent tools, hallucination checking, rule-based eval и LLM-as-judge. Результаты на основном eval подтверждают корректность pipeline, а challenge set демонстрирует ограничения и направления улучшения.
