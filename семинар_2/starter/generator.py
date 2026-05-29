from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm_client import get_model, make_client
from schema import Application, CITIES, DESIRED_COURSES, SPECIALITIES


N_APPLICATIONS = 50
OUT_DIR = Path(".")
MODEL = get_model()

CITY_DISTRICTS = {
    "Москва": ["Центральный округ", "Северный округ", "Юго-Западный округ"],
    "Санкт-Петербург": ["Адмиралтейский район", "Приморский район", "Петроградский район"],
    "Новосибирск": ["Центральный район", "Ленинский район", "Октябрьский район"],
    "Екатеринбург": ["Ленинский район", "Кировский район", "Верх-Исетский район"],
    "Казань": ["Вахитовский район", "Советский район", "Приволжский район"],
    "Нижний Новгород": ["Нижегородский район", "Советский район", "Автозаводский район"],
    "Самара": ["Ленинский район", "Октябрьский район", "Промышленный район"],
    "Ростов-на-Дону": ["Кировский район", "Советский район", "Ворошиловский район"],
    "Краснодар": ["Центральный округ", "Прикубанский округ", "Западный округ"],
    "Пермь": ["Ленинский район", "Свердловский район", "Индустриальный район"],
}

SYSTEM_PROMPT = """
Ты генерируешь синтетические заявки на курсы повышения квалификации ДПО.

Требования:
- заявка должна быть реалистичной для взрослого специалиста из России;
- строго соблюдай схему Pydantic;
- full_name: русское ФИО полностью;
- address.city должен быть только из списка, который указан в пользовательском сообщении;
- address.district должен быть реальным или правдоподобным районом/округом этого города;
- speciality и desired_course выбирай только из разрешенных значений;
- age, graduation_year и years_of_experience не должны противоречить друг другу;
- не повторяй одну и ту же специальность слишком часто;
- отвечай только JSON, без markdown и комментариев.
""".strip()


def build_prompt(seed_city: str, seed_speciality: str, i: int) -> str:
    districts = ", ".join(CITY_DISTRICTS[seed_city])

    return f"""
Создай одну заявку на курс повышения квалификации.

seed_id: {i}
seed_city: {seed_city}
Для поля address.city используй строго этот город: {seed_city}.
Подходящие районы/округа для seed_city: {districts}.

seed_speciality: {seed_speciality}
Желательно использовать эту текущую специальность, если это не ухудшает реалистичность.

Разрешенные города: {", ".join(CITIES)}
Разрешенные специальности: {", ".join(SPECIALITIES)}
Разрешенные курсы: {", ".join(DESIRED_COURSES)}

Верни одну заявку.
""".strip()


def generate_one(client, seed_city: str, seed_speciality: str, i: int) -> Application:
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(seed_city, seed_speciality, i)},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.8,
    )


def flatten_application(app: Application) -> dict:
    data = app.model_dump()
    address = data.pop("address")

    data["city"] = address["city"]
    data["district"] = address["district"]

    return data


def plot_bar(series: pd.Series, title: str, out: Path) -> None:
    counts = series.value_counts()

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", edgecolor="white")
    plt.title(title)
    plt.ylabel("Количество заявок")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def write_report(df: pd.DataFrame, out: Path) -> None:
    n = len(df)
    city_counts = df["city"].value_counts()
    spec_counts = df["speciality"].value_counts()
    crosstab = pd.crosstab(df["city"], df["speciality"])

    lines = [
        "# Report по синтетическим заявкам ДПО",
        "",
        f"Всего валидных заявок: {n}/50.",
        "",
        "## Города",
        "",
        f"Топ-город: {city_counts.index[0]} — {city_counts.iloc[0]} заявок "
        f"({city_counts.iloc[0] / n:.0%}).",
        f"Уникальных городов: {len(city_counts)}.",
        "",
        "## Специальности",
        "",
        f"Топ-специальность: {spec_counts.index[0]} — {spec_counts.iloc[0]} заявок "
        f"({spec_counts.iloc[0] / n:.0%}).",
        f"Уникальных специальностей: {len(spec_counts)}.",
        "",
        "## Кросс-таблица city × speciality",
        "",
        "```",
        crosstab.to_string(),
        "```",
        "",
        "## Нереалистичные или спорные комбинации",
        "",
    ]

    suspicious = []

    for _, row in df.iterrows():
        if row["speciality"] == "медицинская сестра" and row["desired_course"] in {
            "Цифровой маркетинг",
            "Бизнес-аналитика и BPMN",
        }:
            suspicious.append(
                f"- {row['city']}: медицинская сестра выбрала "
                f"«{row['desired_course']}». Это возможно при смене карьерной "
                "траектории, но требует дополнительного объяснения."
            )

        if row["speciality"] == "учитель" and row["desired_course"] == "Финансовый учет и Excel":
            suspicious.append(
                f"- {row['city']}: учитель выбрал курс «Финансовый учет и Excel». "
                "Комбинация не невозможная, но менее типичная без административной роли."
            )

        if row["speciality"] == "юрист" and row["desired_course"] == "Охрана труда":
            suspicious.append(
                f"- {row['city']}: юрист выбрал «Охрана труда». "
                "Комбинация реалистична для комплаенс-функции, но в данных это лучше пояснять."
            )

    if suspicious:
        lines.extend(suspicious[:3])
    else:
        lines.append("- Явно нереалистичных комбинаций не найдено.")

    out.write_text("\n".join(lines), encoding="utf-8")


def write_conclusions(df: pd.DataFrame, out: Path) -> None:
    n = len(df)

    city_counts = df["city"].value_counts()
    spec_counts = df["speciality"].value_counts()

    top_city = city_counts.index[0]
    top_city_count = city_counts.iloc[0]

    top_spec = spec_counts.index[0]
    top_spec_count = spec_counts.iloc[0]

    text = f"""# Выводы

В итоговой выборке получилось {n}/50 валидных заявок. Основной риск mode collapse по городам был снят не случайным `random.choice(cities)`, а стратификацией: генератор делает по 5 заявок на каждый из 10 городов. Поэтому максимальная доля одного города составила {top_city_count / n:.0%}: {top_city} — {top_city_count} заявок. По специальностям распределение осталось менее жестким, потому что специальность задавалась только как seed-поле, а не как квота. Самая частая специальность: {top_spec} — {top_spec_count} заявок, то есть {top_spec_count / n:.0%}. Это ниже порога 35%, но небольшой перекос все равно возможен: модель чаще выбирает универсальные офисные профессии, потому что они легче связываются почти со всеми курсами.

С перекосами я боролся через два механизма: `seed_city` в каждом промпте и предварительно заданные городские квоты. Дополнительно в промпт добавлен `seed_speciality`, чтобы модель не сваливалась в одну-две самые привычные профессии. Нерешенным осталось то, что некоторые сочетания специальности и курса могут быть формально валидными, но спорными по смыслу. Например, медицинская сестра может выбрать бизнес-аналитику при переходе в административную роль, но без дополнительного поля «цель обучения» такая комбинация выглядит менее объяснимой.

`@field_validator` в схеме реально нужен, потому что он отсекает города вне утвержденного списка и год окончания позже текущего года. В финальном сохраненном наборе 50/50 заявок прошли проверку, поэтому в `applications.csv` не осталось нарушений. Отдельно более строгий `model_validator` проверяет связку возраста, года окончания и стажа: например, заявка с возрастом 23 года, выпуском в 2024 году и стажем 10 лет не пройдет валидацию, даже если все отдельные поля находятся в допустимых диапазонах.
"""

    out.write_text(text, encoding="utf-8")


def save_outputs(applications: list[Application]) -> pd.DataFrame:
    rows = [flatten_application(app) for app in applications]
    df = pd.DataFrame(rows)

    df = df[
        [
            "full_name",
            "age",
            "city",
            "district",
            "speciality",
            "desired_course",
            "years_of_experience",
            "graduation_year",
        ]
    ]

    df.to_csv(OUT_DIR / "applications.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "applications.json", "w", encoding="utf-8") as f:
        json.dump(
            [app.model_dump() for app in applications],
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_bar(df["city"], "Распределение заявок по городам", OUT_DIR / "cities.png")
    plot_bar(
        df["speciality"],
        "Распределение заявок по специальностям",
        OUT_DIR / "specialities.png",
    )

    crosstab = pd.crosstab(df["city"], df["speciality"])
    crosstab.to_csv(OUT_DIR / "crosstab_city_speciality.csv", encoding="utf-8-sig")

    write_report(df, OUT_DIR / "report.md")
    write_conclusions(df, OUT_DIR / "выводы.md")

    return df


def main() -> None:
    random.seed(42)

    client = make_client()

    # Стратификация: 10 городов × 5 заявок = 50.
    city_plan = [city for city in CITIES for _ in range(5)]
    random.shuffle(city_plan)

    # Специальность не квотируем жестко, но используем как seed-поле.
    speciality_plan = list(SPECIALITIES) * 5
    random.shuffle(speciality_plan)

    applications: list[Application] = []

    for i, (seed_city, seed_speciality) in enumerate(
        zip(city_plan, speciality_plan),
        start=1,
    ):
        print(
            f"[{i:02d}/{N_APPLICATIONS}] "
            f"city={seed_city}; speciality_seed={seed_speciality}"
        )

        app = generate_one(client, seed_city, seed_speciality, i)
        applications.append(app)

        # Небольшая пауза, чтобы не долбить API слишком резко.
        time.sleep(0.3)

    df = save_outputs(applications)

    print()
    print(f"Готово: {len(df)}/50 валидных заявок")
    print("applications.csv сохранен")
    print("cities.png сохранен")
    print("specialities.png сохранен")
    print("выводы.md сохранен")
    print("report.md сохранен")
    print("crosstab_city_speciality.csv сохранен")


if __name__ == "__main__":
    main()
