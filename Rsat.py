import pandas as pd
import re

input_file = "final_reading(in).csv"
output_file = "Long_final.csv"

df = pd.read_csv(input_file)

# Clean column names
df.columns = df.columns.str.strip()

wanted = ["Participant Private ID", "Article_ID", "Question_ID", "Summary_Text"]

# Make sure all required columns exist
missing = [c for c in wanted if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in CSV: {missing}\nFound: {df.columns.tolist()}")

# Keep only the columns you want
df = df[wanted].copy()

# Clean values
df["Participant Private ID"] = df["Participant Private ID"].astype(str).str.strip()
df["Article_ID"] = df["Article_ID"].astype(str).str.strip()
df["Question_ID"] = df["Question_ID"].astype(str).str.strip()
df["Summary_Text"] = df["Summary_Text"].astype(str).str.strip()

# Optional: normalize Question_ID to Q# (handles "q1", "Q1_text", etc.)
df["Question_ID"] = (
    df["Question_ID"]
    .str.extract(r"(Q\d+)", expand=False)
    .str.upper()
)

# Drop rows missing key info
df = df.dropna(subset=["Participant Private ID", "Question_ID", "Summary_Text"])
df = df[(df["Participant Private ID"] != "") & (df["Question_ID"] != "") & (df["Summary_Text"] != "")]

# Save
df.to_csv(output_file, index=False)
print(f"Saved: {output_file} | Rows: {len(df)}")

# ---------- Helpers ----------
def tokenize(text: str):
    words = re.split(r"\W+", (text or "").lower())
    return set(w for w in words if w)

def split_sentences(text: str):
    sentences = re.split(r"[.!?]", text or "")
    return [tokenize(s) for s in sentences if s.strip()]

def rsat_score(response_words, passage_sentences):
    all_passage_words = set(word for sent in passage_sentences for word in sent)

    overlap = len(response_words & all_passage_words)
    paraphrasing = overlap / len(all_passage_words) if all_passage_words else 0.0

    elaboration = len(response_words - all_passage_words) / len(all_passage_words) if all_passage_words else 0.0

    effort = len(response_words)
    total_words = len(all_passage_words)

    return paraphrasing, elaboration, effort, total_words

# ---------- Main ----------
def process_responses(passages_file, responses_file, output_file):
    # Passages lookup by Article_ID
    passages = {}
    with open(passages_file, newline="", encoding="utf-8") as pf:
        reader = csv.DictReader(pf)
        for row in reader:
            article_id = (row.get("Article_ID") or "").strip()
            passage = (row.get("passage") or "").strip()
            if article_id and passage:
                passages[article_id] = passage

    with open(responses_file, newline="", encoding="utf-8") as rf, \
         open(output_file, "w", newline="", encoding="utf-8") as wf:

        reader = csv.DictReader(rf)

        fieldnames = [
            "Participant Private ID", "Article_ID", "Question_ID", "Summary_Text",
            "paraphrasing (%)", "elaboration (%)", "effort (words)", "total_effort (%)"
        ]
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            pid = (row.get("Participant Private ID") or "").strip()
            article_id = (row.get("Article_ID") or "").strip()
            qid = (row.get("Question_ID") or "").strip()
            response = (row.get("Summary_Text") or "").strip()

            if not pid or not article_id or not response:
                continue

            if article_id not in passages:
                print(f"Warning: no passage for Article_ID={article_id} (participant {pid}, {qid})")
                continue

            passage_sentences = split_sentences(passages[article_id])
            response_words = tokenize(response)

            paraphrasing, elaboration, effort, total_words = rsat_score(response_words, passage_sentences)
            total_effort = effort / total_words if total_words else 0.0

            writer.writerow({
                "Participant Private ID": pid,
                "Article_ID": article_id,
                "Question_ID": qid,
                "Summary_Text": response,
                "paraphrasing (%)": round(paraphrasing, 3),
                "elaboration (%)": round(elaboration, 3),
                "effort (words)": effort,
                "total_effort (%)": round(total_effort, 3),
            })

    print(f"RSAT scores written to {output_file}")

process_responses("Passages.csv", "Long_final.csv", "rsat2_scored.csv")