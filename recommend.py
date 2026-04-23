import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("products.csv", encoding="utf-8")
df["content"] = df["name"] + " " + df["category"] + " " + df["description"]

# Đảm bảo cột image_url tồn tại
if "image_url" not in df.columns:
    df["image_url"] = ""

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["content"])
similarity_matrix = cosine_similarity(tfidf_matrix)


def normalize_text(text):
    text = str(text).strip().lower()
    text = text.replace("\u0111", "d")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def get_products():
    return [
        {"name": row["name"], "image_url": row.get("image_url", "")}
        for _, row in df.iterrows()
    ]


def get_product_names():
    return df["name"].tolist()


def recommend(product_name, top_n=3):
    normalized_name = normalize_text(product_name)
    matches = df[df["name"].apply(normalize_text) == normalized_name]

    if matches.empty:
        raise ValueError(f"Khong tim thay san pham: {product_name}")

    idx = matches.index[0]
    query_row = df.iloc[idx]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i in scores[1 : top_n + 1]:
        row = df.iloc[i[0]]
        results.append({
            "name": row["name"],
            "image_url": row.get("image_url", ""),
        })

    return {
        "query": {
            "name": query_row["name"],
            "image_url": query_row.get("image_url", ""),
        },
        "recommendations": results,
    }