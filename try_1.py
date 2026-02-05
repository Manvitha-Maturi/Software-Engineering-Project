"""
trace_matrices_only_nltk.py

Runs the SAME pipeline your professor expects, uses NLTK for:
- tokenization
- stopword removal
- stemming
- lemmatization

Pipeline (per variant):
Preprocess -> TF-IDF (with TF squashing) -> Cosine -> MergeSort ranking -> Top-K binarize
Output (per variant):
- Full 60x3 predicted trace matrix (FRi, b1, b2, b3)
- Accuracy ONLY (compared to provided answer set)

FILES (put in same folder as this script):
- requirements-3nfr-60fr.txt
- trace-3nfr-60fr.txt
Run:
  python try_1.py
"""

import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------- NLTK downloads (safe to keep) ----------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # sometimes needed on newer NLTK
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ---------- Files (same folder recommended) ----------
REQ_FILE = "requirements-3nfr-60fr.txt"
TRACE_FILE = "trace-3nfr-60fr.txt"

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


# =========================================================
# Parse requirements + trace
# =========================================================

def parse_requirements(path: str):
    """
    Reads the requirements file and extracts:
    - NFR1..NFR3 lines
    - FR1..FR60 lines
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # NFR lines: NFR1 (Type): ... OR NFR1: ...
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
    """
    Parses answer key rows like:
      FR1,0,1,0
    Returns (trace_fr_ids, y_true) where y_true is (60,3).
    """
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
# Preprocessing (NLTK) for 3 variants
# =========================================================

def normalize_tokens(tokens):
    """
    Keep alphanumeric tokens only, lowercase.
    This keeps your vocabulary clean and consistent.
    """
    clean = []
    for t in tokens:
        t = t.lower()
        if any(ch.isalnum() for ch in t):
            # strip non-alphanum at ends (e.g., commas, periods)
            t = re.sub(r"^\W+|\W+$", "", t)
            if t and re.fullmatch(r"[a-z0-9]+", t):
                clean.append(t)
    return clean


def preprocess_nltk(text: str, variant: int) -> str:
    """
    Variant 1: Tokenization + stopword removal
    Variant 2: + stemming
    Variant 3: + lemmatization
    Returns a single string of tokens separated by spaces.
    """
    # 1) Tokenize using NLTK
    tokens = word_tokenize(text)

    # 2) normalize + lowercase + keep alphanumeric
    tokens = normalize_tokens(tokens)

    # 3) Stopword removal
    tokens = [t for t in tokens if t not in STOPWORDS]

    # 4) Variant-specific
    if variant == 2:
        tokens = [STEMMER.stem(t) for t in tokens]
    elif variant == 3:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    elif variant != 1:
        raise ValueError("variant must be 1, 2, or 3")

    return " ".join(tokens)


# =========================================================
# Merge Sort (descending by score)
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
# TF-IDF + cosine + ranking + binarize + print
# =========================================================

def compute_similarity(fr_proc, nfr_proc):
    """
    TF-IDF:
      - use_idf=True (idf weighting)
      - sublinear_tf=True (TF squashing: 1 + log(tf))
      - norm='l2' (so cosine similarity is meaningful)
    """
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    FR_tfidf = vectorizer.fit_transform(fr_proc)
    NFR_tfidf = vectorizer.transform(nfr_proc)
    return cosine_similarity(FR_tfidf, NFR_tfidf)  # (60,3)


def rank_by_nfr(S, fr_ids):
    rankings = {}
    for j in range(3):
        pairs = [(i, float(S[i, j])) for i in range(len(fr_ids))]
        sorted_pairs = merge_sort_pairs(pairs)
        rankings[j] = [(fr_ids[i], score) for (i, score) in sorted_pairs]
    return rankings


def binarize_topk(rankings, fr_ids, top_k=10):
    """
    Convert rankings into a 60x3 binary matrix.
    For each NFR column, mark top_k FRs as 1.
    """
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


# =========================================================
# MAIN
# =========================================================

def main():
    nfr_ids, nfr_texts, fr_ids, fr_texts = parse_requirements(REQ_FILE)
    trace_fr_ids, y_true = parse_trace(TRACE_FILE)

    # Align answer-key rows to FR order in requirements file (just in case)
    if fr_ids != trace_fr_ids:
        idx_map = {fid: i for i, fid in enumerate(trace_fr_ids)}
        y_true = np.array([y_true[idx_map[fid]] for fid in fr_ids], dtype=int)

    TOP_K = 10  # keep as-is unless your prof specifies another K

    for variant in [1, 2, 3]:
        fr_proc = [preprocess_nltk(t, variant) for t in fr_texts]
        nfr_proc = [preprocess_nltk(t, variant) for t in nfr_texts]

        S = compute_similarity(fr_proc, nfr_proc)
        rankings = rank_by_nfr(S, fr_ids)
        y_pred = binarize_topk(rankings, fr_ids, top_k=TOP_K)

        print("\n" + "=" * 40)
        print(f"VARIANT {variant} TRACE MATRIX (NLTK)")
        print("=" * 40)
        print_full_matrix(fr_ids, nfr_ids, y_pred)

        acc = accuracy_score(y_true.flatten(), y_pred.flatten())
        print(f"\nAccuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
