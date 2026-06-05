from pathlib import Path

import pandas as pd


INPUT_CSV = Path("input/reviews.csv")
OUT_DIR = Path("input")
N_REVIEWS = 40
N_BATCHES = 5

APP_ID = "com.anydo"
APP_NAME = "Any.do"

df = pd.read_csv(INPUT_CSV)

required = {"reviewId", "content", "score", "at", "appId"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"В CSV не хватает колонок: {missing}")

df = df.dropna(subset=["content", "score", "at", "appId"])
df = df[df["content"].astype(str).str.len() >= 30]

# Берем одно приложение, чтобы предметная область была цельной.
app_df = df[df["appId"] == APP_ID].copy()

if len(app_df) < N_REVIEWS:
    raise ValueError(f"Для appId={APP_ID} найдено меньше {N_REVIEWS} отзывов")

# Берем смесь плохих, средних и хороших отзывов, чтобы анализ был содержательным.
parts = []
for score, n in [(1, 10), (2, 8), (3, 7), (4, 7), (5, 8)]:
    chunk = app_df[app_df["score"] == score].head(n)
    parts.append(chunk)

sample = pd.concat(parts, ignore_index=True)

if len(sample) < N_REVIEWS:
    sample = app_df.head(N_REVIEWS)

sample = sample.head(N_REVIEWS).reset_index(drop=True)

for batch_idx in range(N_BATCHES):
    batch = sample.iloc[batch_idx * 8 : (batch_idx + 1) * 8]
    out_path = OUT_DIR / f"reviews_batch_{batch_idx + 1}.txt"

    blocks = []

    for i, row in batch.iterrows():
        review_id = f"R{i + 1:03d}"
        rating = int(row["score"])
        date = str(row["at"])[:10]
        text = str(row["content"]).replace("\n", " ").strip()

        block = f"""Review ID: {review_id}
App: {APP_NAME}
Source: Google Play
Rating: {rating}
Date: {date}
Text: {text}"""

        blocks.append(block)

    out_path.write_text("\n\n".join(blocks), encoding="utf-8")

print(f"Готово: создано {N_BATCHES} файлов по 8 отзывов")
print("Файлы:")
for path in sorted(OUT_DIR.glob("reviews_batch_*.txt")):
    print("-", path)
