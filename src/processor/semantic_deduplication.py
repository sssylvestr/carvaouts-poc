#!pip install -U sentence-transformers faiss-gpu-cu12 [works on Linux with NVIDIA GPU, will not work on Mac/Windows, careful!]
# --------------------------------------------------------------------
# 4)  Semantic dedup that handles *long* (≤ 2k-word) segments
# --------------------------------------------------------------------
import numpy as np, pandas as pd
import faiss
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

TEXT_COL = "segments"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CHUNK_TOKENS = 350  # window length
STRIDE = 300  # overlap: CHUNK_TOKENS – STRIDE  (= 50 here)
COS_THRESHOLD = 0.95
BATCH_CHUNKS = 512

model = SentenceTransformer(MODEL_NAME, device="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
DIM = model.get_sentence_embedding_dimension()


# ───────────────────  helper: chunk→embed→mean ───────────────────────
def embed_long(text: str) -> np.ndarray:
    """Mean-pool E5 embeddings of overlapping CHUNK_TOKENS windows."""
    # 1) tokenise once
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    # 2) split into overlapping windows
    slices = [ids[i : i + CHUNK_TOKENS] for i in range(0, len(ids), STRIDE)] or [
        ids
    ]  # edge-case: empty → keep one slice

    # 3) decode back to strings for ST encoder
    windows = tokenizer.batch_decode(slices, skip_special_tokens=True)

    # 4) encode windows in smaller sub-batches to fit GPU
    embs, out = [], None
    for s in range(0, len(windows), BATCH_CHUNKS):
        sub = windows[s : s + BATCH_CHUNKS]
        sub_emb = model.encode(
            sub,
            batch_size=len(sub),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embs.append(sub_emb)
    out = np.vstack(embs).mean(axis=0)
    # 5) re-normalise pooled vector → cosine ready
    out /= np.linalg.norm(out) + 1e-12
    return out.astype("float32")


def deduplicate(df, text_col="segments", threshold=0.95):
    """
    Deduplicate long segments in a DataFrame using semantic similarity.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_col (str): Column name containing the text to deduplicate.
        threshold (float): Cosine similarity threshold for deduplication.

    Returns:
        pd.DataFrame: DataFrame with deduplicated segments.
    """
    # Ensure the column exists
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    texts = df[text_col].tolist()
    doc_embs = np.empty((len(texts), DIM), dtype="float32")

    for i, txt in enumerate(tqdm(texts, desc="Encoding long docs")):
        doc_embs[i] = embed_long(txt)

    # ────────────────────  FAISS search & clustering  ────────────────────
    index = faiss.IndexFlatIP(DIM)
    index.add(doc_embs)

    D, I = index.search(doc_embs, 2)

    N = len(df)
    pairs = []
    for i, (sim, j) in tqdm(enumerate(zip(D[:, 1], I[:, 1]))):
        j = int(j)
        if 0 <= j < N and i != j and sim >= COS_THRESHOLD:
            pairs.append((i, j))

    parent = list(range(len(df)))
    size = [1] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in tqdm(pairs):
        union(a, b)

    clusters = {}
    for idx in tqdm(range(len(df))):
        clusters.setdefault(find(idx), []).append(idx)

    keep_idx = {max(ids, key=lambda i: len(df.at[i, TEXT_COL])) for ids in clusters.values()}

    df_semantic = df.loc[sorted(keep_idx)].reset_index(drop=True)
    print(f"Semantic dedup kept {len(df_semantic)} / {len(df)} rows")

    return df_semantic
