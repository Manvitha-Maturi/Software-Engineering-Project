import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# -----------------------
# Files (must be in same folder OR set full paths)
# -----------------------
REQ_FILE = "requirements-3nfr-60fr.txt"
TRACE_FILE = "trace-3nfr-60fr.txt"

STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


# =========================================================
# Parse requirements + trace
# =========================================================

def parse_requirements(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    nfr_matches = re.findall(r"^(NFR\d+).*?:\s*(.+)$", text, flags=re.MULTILINE)
    nfr_ids, nfr_texts = [], []
    for nid, desc in nfr_matches:
        nfr_ids.append(nid.strip())
        nfr_texts.append(desc.strip())

    fr_matches = re.findall(r"^(FR\d+):\s*(.+)$", text, flags=re.MULTILINE)
    fr_ids, fr_texts = [], []
    for fid, desc in fr_matches:
        fr_ids.append(fid.strip())
        fr_texts.append(desc.strip())

    if len(nfr_ids) != 3:
        raise ValueError(f"Expected 3 NFRs, found {len(nfr_ids)}")
    if len(fr_ids) != 60:
        raise ValueError(f"Expected 60 FRs, found {len(fr_ids)}")

    return nfr_ids, nfr_texts, fr_ids, fr_texts


def parse_trace(path: str):
    rows = []
    fr_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Bad line in trace file: {line}")
            fr_ids.append(parts[0])
            rows.append(list(map(int, parts[1:])))

    y_true = np.array(rows, dtype=int)
    if y_true.shape != (60, 3):
        raise ValueError(f"Expected trace matrix shape (60,3), got {y_true.shape}")

    return fr_ids, y_true


# =========================================================
# Preprocessing variants
# =========================================================

def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

def preprocess(text: str, variant: int) -> str:
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]

    if variant == 2:
        tokens = [STEMMER.stem(t) for t in tokens]
    elif variant == 3:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    elif variant != 1:
        raise ValueError("variant must be 1, 2, or 3")

    return " ".join(tokens)


# =========================================================
# Merge sort (descending by score)
# =========================================================

def merge_sort_pairs(pairs):
    if len(pairs) <= 1:
        return pairs
    mid = len(pairs) // 2
    left = merge_sort_pairs(pairs[:mid])
    right = merge_sort_pairs(pairs[mid:])
    return merge_desc(left, right)

def merge_desc(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] >= right[j][1]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


# =========================================================
# TF-IDF + cosine (with TF squashing)
# =========================================================

def similarity_matrix(fr_proc, nfr_proc):
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True  # TF squashing (1 + log(tf))
    )
    FR_tfidf = vectorizer.fit_transform(fr_proc)
    NFR_tfidf = vectorizer.transform(nfr_proc)
    return cosine_similarity(FR_tfidf, NFR_tfidf)  # (60,3)


def rank_by_nfr(S, fr_ids, nfr_count=3):
    # returns dict: nfr_index -> list of (fr_id, score) sorted desc
    rankings = {}
    for j in range(nfr_count):
        pairs = [(i, float(S[i, j])) for i in range(len(fr_ids))]
        sorted_pairs = merge_sort_pairs(pairs)
        rankings[j] = [(fr_ids[i], score) for (i, score) in sorted_pairs]
    return rankings


def binarize_from_rankings(rankings, fr_ids, top_k=10):
    # top-k binary decision (matches your current run)
    fr_index = {fid: i for i, fid in enumerate(fr_ids)}
    y_pred = np.zeros((len(fr_ids), 3), dtype=int)

    for j in range(3):
        for fr_id, _score in rankings[j][:top_k]:
            y_pred[fr_index[fr_id], j] = 1

    return y_pred


def print_full_matrix(fr_ids, nfr_ids, y_pred):
    print("FR," + ",".join(nfr_ids))
    for i, fr_id in enumerate(fr_ids):
        row = [fr_id] + list(map(int, y_pred[i].tolist()))
        print(",".join(map(str, row)))


def main():
    nfr_ids, nfr_texts, fr_ids, fr_texts = parse_requirements(REQ_FILE)
    trace_fr_ids, y_true = parse_trace(TRACE_FILE)

    # align y_true to FR order if needed
    if fr_ids != trace_fr_ids:
        idx_map = {fid: i for i, fid in enumerate(trace_fr_ids)}
        y_true = np.array([y_true[idx_map[fid]] for fid in fr_ids], dtype=int)

    TOP_K = 10  # keep same as your run; you can change to 5 if prof wants

    for variant in [1, 2, 3]:
        fr_proc = [preprocess(t, variant) for t in fr_texts]
        nfr_proc = [preprocess(t, variant) for t in nfr_texts]

        S = similarity_matrix(fr_proc, nfr_proc)
        rankings = rank_by_nfr(S, fr_ids, nfr_count=3)
        y_pred = binarize_from_rankings(rankings, fr_ids, top_k=TOP_K)

        # Print ONLY what you want
        print("\n" + "=" * 40)
        print(f"VARIANT {variant} TRACE MATRIX")
        print("=" * 40)
        print_full_matrix(fr_ids, nfr_ids, y_pred)

        acc = accuracy_score(y_true.flatten(), y_pred.flatten())
        print(f"\nAccuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
