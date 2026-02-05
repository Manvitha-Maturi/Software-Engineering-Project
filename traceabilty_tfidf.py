import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# NLTK for stemming + lemmatization
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# -----------------------
# FILE PATHS (your files)
# -----------------------
REQ_FILE = "requirements-3nfr-60fr.txt"
TRACE_FILE = "trace-3nfr-60fr.txt"

# -----------------------
# NLP TOOLS
# -----------------------
STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# =========================================================
# 1) PARSING REQUIREMENTS + TRACE (ANSWER KEY)
# =========================================================

def parse_requirements(path: str):
    """
    Parses:
      NFR1 ... NFR3
      FR1 ... FR60
    from the provided txt format.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # NFR lines: NFR1 (Type): ... OR NFR1: ...
    nfr_matches = re.findall(r"^(NFR\d+).*?:\s*(.+)$", text, flags=re.MULTILINE)
    nfr_ids, nfr_texts = [], []
    for nfr_id, nfr_desc in nfr_matches:
        nfr_ids.append(nfr_id.strip())
        nfr_texts.append(nfr_desc.strip())

    # FR lines: FR1: ...
    fr_matches = re.findall(r"^(FR\d+):\s*(.+)$", text, flags=re.MULTILINE)
    fr_ids, fr_texts = [], []
    for fr_id, fr_desc in fr_matches:
        fr_ids.append(fr_id.strip())
        fr_texts.append(fr_desc.strip())

    if len(nfr_ids) != 3:
        raise ValueError(f"Expected 3 NFRs, found {len(nfr_ids)}")
    if len(fr_ids) != 60:
        raise ValueError(f"Expected 60 FRs, found {len(fr_ids)}")

    return nfr_ids, nfr_texts, fr_ids, fr_texts


def parse_trace(path: str):
    """
    Parses answer set rows like:
      FR1,0,1,0
    Returns:
      trace_fr_ids: list[str] length 60
      y_true: np.ndarray shape (60,3)
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
            fr_id = parts[0]
            vals = list(map(int, parts[1:]))
            fr_ids.append(fr_id)
            rows.append(vals)

    y_true = np.array(rows, dtype=int)
    if y_true.shape != (60, 3):
        raise ValueError(f"Expected trace matrix shape (60,3), got {y_true.shape}")
    return fr_ids, y_true


# =========================================================
# 2) PREPROCESSING VARIANTS (as your assignment specifies)
#    Variant 1: Tokenization + stopword removal
#    Variant 2: + stemming
#    Variant 3: + lemmatization
# =========================================================

def tokenize(text: str) -> list[str]:
    text = text.lower()
    # split hyphens, keep alphanumeric
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

    # TF-IDF vectorizer expects strings, so join tokens
    return " ".join(tokens)


# =========================================================
# 3) TF-IDF (slide-aligned)
#    - Use tf-idf weighting
#    - Use TF "squashing" (sublinear TF): tf = 1 + log(tf)
# =========================================================

def compute_similarity(fr_texts_proc, nfr_texts_proc):
    """
    Fit TF-IDF on FRs (documents), transform FRs and NFRs (queries).
    Uses sublinear_tf=True to match "squash TF" idea from slides.
    """
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True   # <-- THIS is the TF squashing from slides
    )

    FR_tfidf = vectorizer.fit_transform(fr_texts_proc)
    NFR_tfidf = vectorizer.transform(nfr_texts_proc)

    # Cosine similarity (as in slides)
    S = cosine_similarity(FR_tfidf, NFR_tfidf)  # shape (60,3)
    return S


# =========================================================
# 4) SORTING (slide-aligned)
#    Your assignment mentions Merge Sort or Quick Sort.
#    We'll implement Merge Sort for ranking by similarity.
# =========================================================

def merge_sort_pairs(pairs):
    """
    Stable merge sort for list of tuples: (fr_index, score)
    Sort descending by score.
    """
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
        # Descending by score
        if left[i][1] >= right[j][1]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def rank_FRs_for_each_NFR(S, fr_ids, nfr_ids, topn=10):
    """
    Uses merge sort (not numpy sort) to rank FRs per NFR.
    """
    rankings = {}

    for j, nfr_id in enumerate(nfr_ids):
        pairs = [(i, float(S[i, j])) for i in range(len(fr_ids))]
        sorted_pairs = merge_sort_pairs(pairs)
        rankings[nfr_id] = [(fr_ids[i], score) for (i, score) in sorted_pairs]

        print(f"\nTop {topn} FRs for {nfr_id} (sorted by Merge Sort):")
        for fr_id, score in rankings[nfr_id][:topn]:
            print(f"  {fr_id}  score={score:.4f}")

    return rankings


# =========================================================
# 5) CONVERT SCORES -> BINARY TRACE MATRIX
#    (this is needed to compare with your answer set)
# =========================================================

def binarize_links_from_rankings(rankings, fr_ids, nfr_ids, method="topk", top_k=10, threshold=0.15):
    """
    Builds a 60x3 binary matrix y_pred from ranked lists.

    method:
      - "topk": mark top_k as 1
      - "threshold": mark score>=threshold as 1
      - "hybrid": top_k AND score>=threshold
    """
    fr_index = {fid: i for i, fid in enumerate(fr_ids)}
    nfr_index = {nid: j for j, nid in enumerate(nfr_ids)}

    y_pred = np.zeros((len(fr_ids), len(nfr_ids)), dtype=int)

    for nid in nfr_ids:
        ranked_list = rankings[nid]
        j = nfr_index[nid]

        if method == "topk":
            for fr_id, _score in ranked_list[:top_k]:
                y_pred[fr_index[fr_id], j] = 1

        elif method == "threshold":
            for fr_id, score in ranked_list:
                if score >= threshold:
                    y_pred[fr_index[fr_id], j] = 1

        elif method == "hybrid":
            for fr_id, score in ranked_list[:top_k]:
                if score >= threshold:
                    y_pred[fr_index[fr_id], j] = 1

        else:
            raise ValueError("method must be 'topk', 'threshold', or 'hybrid'")

    return y_pred


# =========================================================
# 6) EVALUATION (compare to given answer set)
# =========================================================

def evaluate(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()

    acc = accuracy_score(yt, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    return acc, prec, rec, f1


def print_trace_matrix(fr_ids, nfr_ids, y_pred, max_rows=12):
    """
    Prints rows in the professor-style format:
      FRi, b1, b2, b3
    """
    print("\nPredicted trace matrix (sample):")
    print("FR," + ",".join(nfr_ids))
    for i in range(min(max_rows, len(fr_ids))):
        row = [fr_ids[i]] + list(map(int, y_pred[i].tolist()))
        print(",".join(map(str, row)))


# =========================================================
# MAIN: RUN ALL 3 VARIANTS
# =========================================================

def main():
    nfr_ids, nfr_texts, fr_ids, fr_texts = parse_requirements(REQ_FILE)
    trace_fr_ids, y_true = parse_trace(TRACE_FILE)

    # Align y_true to FR order if needed
    if fr_ids != trace_fr_ids:
        print("WARNING: FR order mismatch between requirements file and trace file. Aligning by FR_id...")
        idx_map = {fid: i for i, fid in enumerate(trace_fr_ids)}
        y_true = np.array([y_true[idx_map[fid]] for fid in fr_ids], dtype=int)

    # Choose binarization rule (you can tune to match your course’s expected linking)
    BIN_METHOD = "topk"      # "topk" or "threshold" or "hybrid"
    TOP_K = 10               # typical values: 5 or 10
    THRESH = 0.15            # typical: 0.10–0.20

    summary = []

    for variant in [1, 2, 3]:
        print("\n" + "=" * 90)
        print(f"VARIANT {variant}")
        print("=" * 90)

        # A) Preprocessing outputs (show sample, as slides request “output after each step”)
        fr_proc = [preprocess(t, variant) for t in fr_texts]
        nfr_proc = [preprocess(t, variant) for t in nfr_texts]

        print("\nPreprocessing sample (before -> after):")
        print("NFR1 BEFORE:", nfr_texts[0])
        print("NFR1 AFTER: ", nfr_proc[0])
        print("FR1  BEFORE:", fr_texts[0])
        print("FR1  AFTER: ", fr_proc[0])

        # B) TF-IDF + cosine similarity
        S = compute_similarity(fr_proc, nfr_proc)

        # C) Sorting (Merge Sort) + show top results
        rankings = rank_FRs_for_each_NFR(S, fr_ids, nfr_ids, topn=8)

        # D) Binary trace matrix (0/1)
        y_pred = binarize_links_from_rankings(
            rankings, fr_ids, nfr_ids,
            method=BIN_METHOD, top_k=TOP_K, threshold=THRESH
        )

        # Print sample output
        print_trace_matrix(fr_ids, nfr_ids, y_pred, max_rows=12)

        # E) Compare to answer set
        acc, prec, rec, f1 = evaluate(y_true, y_pred)
        print("\nEvaluation vs answer key:")
        print(f"  Binarization: method={BIN_METHOD}, top_k={TOP_K}, threshold={THRESH}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")

        summary.append((variant, acc, prec, rec, f1))

    print("\n" + "=" * 90)
    print("SUMMARY (Variant -> metrics)")
    for variant, acc, prec, rec, f1 in summary:
        print(f"Variant {variant}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
