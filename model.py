import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bert_score import score as bert_score
from rouge_score import rouge_scorer
#df = pd.read_csv("Diverse_HR_Employee_Dataset_200.csv")
# ====== Heuristic Hints ======
SKILL_HINTS = ["python", "sql", "ml", "ai", "docker", "power bi", "java", "tableau"]
CERT_HINTS = ["aws", "azure", "certified", "certification", "google", "openai", "huggingface"]
FEEDBACK_HINTS = ["great", "support", "mentor", "help", "team", "improve", "feedback"]
EXPERIENCE_RANGE = (0, 40)
# ====== Alias Definitions ======
COLUMN_ALIASES = {
    "skills": ["skills", "skillset", "technical skills", "key skills"],
    "certifications": ["certifications", "relevant qualifications", "accreditations", "certs","relevant certifications"],
    "experience": ["experience (years)", "years of experience", "exp", "total experience"],
    "peer_feedback": ["peer feedback", "peer review", "team feedback", "peer comments","peer reviews"],
    "manager_feedback": [
        "manager feedback", "supervisor feedback", "performance feedback",
        "manager performance review", "supervisor performance review",
        "manager performance feedback", "supervisor performance feedback",
        "performance review", "performance feedback", "performace reviews",
        "manager performance reviews", "supervisor performance reviews"
    ]

}
# ====== Column Detection ======
def detect_column(df, aliases, check_type):
    for alias in aliases:
        for col in df.columns:
            if alias.lower() in col.lower():
                return col, "alias"
    sample = df.head(10)

    if check_type == "experience":
        for col in df.columns:
            try:
                values = pd.to_numeric(sample[col], errors='coerce').dropna()
                if len(values) >= 5 and values.between(*EXPERIENCE_RANGE).mean() > 0.7:
                    return col, "heuristic"
            except:
                continue
    elif check_type == "skills":
        for col in df.columns:
            matches = sample[col].dropna().apply(lambda x: any(term in str(x).lower() for term in SKILL_HINTS))
            if matches.sum() >= 5:
                return col, "heuristic"
    elif check_type == "certifications":
        for col in df.columns:
            matches = sample[col].dropna().apply(lambda x: any(term in str(x).lower() for term in CERT_HINTS))
            if matches.sum() >= 5:
                return col, "heuristic"
    elif check_type in ["peer_feedback", "manager_feedback"]:
        for col in df.columns:
            text_check = sample[col].dropna().apply(lambda x: len(str(x).split()) > 4)
            keyword_check = sample[col].dropna().apply(lambda x: any(k in str(x).lower() for k in FEEDBACK_HINTS))
            if text_check.sum() >= 5 and keyword_check.sum() >= 3:
                return col, "heuristic"
    return None, "not found"
def detect_all_columns(df):
    results = {}
    for key, aliases in COLUMN_ALIASES.items():
        col, method = detect_column(df, aliases, key)
        results[key] = {"column": col, "method": method}
    return results
# ====== Sentiment Analysis ======
def sentiment_score(text, analyzer):
    result = analyzer(text[:512])[0]
    return result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
# ====== Evaluation Metrics ======
def evaluate_with_bert_rouge(results_df, query_text):
    candidate_texts = results_df["embedding_text"].tolist()
    reference_texts = [query_text] * len(candidate_texts)
    P, R, F1 = bert_score(candidate_texts, reference_texts, lang="en", model_type="roberta-large", verbose=False)
    bert_scores = F1.tolist()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(query_text, cand) for cand in candidate_texts]
    rouge1 = [r["rouge1"].fmeasure for r in rouge_scores]
    rougeL = [r["rougeL"].fmeasure for r in rouge_scores]

    return bert_scores, rouge1, rougeL
# ====== Recommendation Engine ======
def hybrid_recommendation(df, query, top_k=10):
    column_map = detect_all_columns(df)
    col = {k: v["column"] for k, v in column_map.items()}
    assert all(col.values()), "Missing required columns in dataset."

    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device = 0)

    # Build embedding text
    def build_text(row):
        return (
            f"{row['Employee Name']} is a {row['Job Title']} with {row[col['experience']]} years of experience. "
            f"Skills: {row[col['skills']]}. Certifications: {row[col['certifications']]}. "
            f"Peer feedback: {row[col['peer_feedback']]}. Manager feedback: {row[col['manager_feedback']]}."
        )

    df["embedding_text"] = df.apply(build_text, axis=1)
    embeddings = embedder.encode(df["embedding_text"].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    df["Peer Sentiment"] = df[col["peer_feedback"]].apply(lambda x: sentiment_score(x, sentiment_analyzer))
    df["Manager Sentiment"] = df[col["manager_feedback"]].apply(lambda x: sentiment_score(x, sentiment_analyzer))
    df["Feedback Score"] = 0.7 * df["Manager Sentiment"] + 0.3 * df["Peer Sentiment"]

    def extract_terms(column):
        terms = set()
        for entry in df[column].dropna():
            for term in str(entry).split(","):
                terms.add(term.strip().lower())
        return terms
    #Extract skills and certs from dataset
    skills_set = extract_terms(col["skills"])
    certs_set = extract_terms(col["certifications"])
    # Step: Extract location values from dataset
    locations_set = set(str(loc).strip().lower() for loc in df["Location"].dropna().unique())

    ## parsing the query to extract skills, certs, location and exp
    def parse_query(query):
        q = query.lower()
        skills = {s for s in skills_set if s in q}
        certs = {c for c in certs_set if c in q}
        location = next((loc for loc in locations_set if loc in q), None)
        exp_match = re.search(r"(\d+)\s*\+?\s*years?", q)
        min_exp = int(exp_match.group(1)) if exp_match else 0
        return list(skills), list(certs), min_exp, location

    required_skills, preferred_certs, min_exp, location = parse_query(query)
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), k=top_k * 2)
    candidates = df.iloc[indices[0]].copy()
    candidates = candidates[candidates[col["experience"]] >= min_exp]
    if location:
      candidates = candidates[candidates["Location"].str.lower() == location]

    candidates["skill_match"] = candidates[col["skills"]].str.lower().apply(
        lambda s: sum(skill in s for skill in required_skills) / max(len(required_skills), 1)
    )
    candidates["cert_match"] = candidates[col["certifications"]].str.lower().apply(
        lambda c: sum(cert in c for cert in preferred_certs) / max(len(preferred_certs), 1)
    )
    max_exp = df[col["experience"]].max()
    candidates["exp_score"] = candidates[col["experience"]] / max_exp

    candidates["final_score"] = (
        0.4 * candidates["skill_match"] +
        0.3 * candidates["cert_match"] +
        0.2 * candidates["Feedback Score"] +
        0.1 * candidates["exp_score"]
    )

    top_candidates = candidates.sort_values(by="final_score", ascending=False).head(top_k).copy()
    top_candidates["Rank"] = range(1, len(top_candidates) + 1)

    bert_f1, rouge1, rougeL = evaluate_with_bert_rouge(top_candidates, query)
    top_candidates["BERTScore F1"] = bert_f1
    top_candidates["ROUGE-1 F"] = rouge1
    top_candidates["ROUGE-L F"] = rougeL

    return top_candidates[[
        "Rank", "Employee Name", "Employee ID", "Job Title", col["skills"], col["certifications"],
        col["experience"], col["peer_feedback"], col["manager_feedback"],
        "Peer Sentiment", "Manager Sentiment", "Feedback Score",
        "skill_match", "cert_match", "exp_score", "final_score",
        "BERTScore F1", "ROUGE-1 F", "ROUGE-L F"
    ]]
