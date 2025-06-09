import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from transformers import AutoTokenizer

from typing import List

logging.basicConfig(level=logging.INFO)

PREFIX = "query: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic deduplication with multilingual‑e5 embeddings and FAISS")
    parser.add_argument("--input","-i", required=True, help="Input CSV/Parquet file")
    parser.add_argument("--output","-o", required=True, help="Output file path (CSV/Parquet)")
    parser.add_argument("--column", "-c", default="segment", help="Name of the text column")
    parser.add_argument("--chunk_tokens", type=int, default=350, help="Window length")
    parser.add_argument("--stride", type=int, default=300, help="Token step (overlap)")
    parser.add_argument("--threshold", type=float, default=0.95, help="Cosine threshold")
    parser.add_argument("--batch_chunks", type=int, default=128, help="Batch size for encoding")
    parser.add_argument(
        "--model",
        "-m",
        default="intfloat/multilingual-e5-small",
        help="Sentence‑Transformer model name",
    )

    parser.add_argument("--device", default="cuda", help="Device for Sentence‑Transformer model")
    return parser.parse_args()


def sliding_windows(ids: np.ndarray, chunk: int, stride: int) -> list[list[int]]:
    """Return overlapping windows of token ids (≤chunk tokens each)."""
    if len(ids) <= chunk:
        return [ids.tolist()]

    from numpy.lib.stride_tricks import sliding_window_view

    view = sliding_window_view(ids, chunk)[::stride]
    return [row.tolist() for row in view]


def embed_all_docs(
    df: pd.DataFrame,
    text_col: str,
    tokenizer: AutoTokenizer,
    model: SentenceTransformer,
    chunk_tokens: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    """Encode every overlapping window once, aggregate by mean‑pooling per doc."""
    dim = model.get_sentence_embedding_dimension()

    # 1) Build flat window list
    all_windows: List[str] = []
    owners: List[int] = []
    for doc_id, text in enumerate(tqdm(df[text_col], desc="chunking")):
        token_ids = np.array(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int32)  #type: ignore
        slices = sliding_windows(token_ids, chunk_tokens, stride)
        decoded = tokenizer.batch_decode(slices, skip_special_tokens=True) #type: ignore
        all_windows.extend(PREFIX + t for t in decoded)
        owners.extend([doc_id] * len(decoded))

    owners_np = np.array(owners, dtype=np.int32)
    sums = np.zeros((len(df), dim), dtype="float32") #type: ignore
    counts = np.zeros(len(df), dtype=np.int32)

    # 2) Encode in global batches
    for start in tqdm(range(0, len(all_windows), batch_size), desc="encoding"):
        texts_batch = all_windows[start : start + batch_size]
        vecs = model.encode(
            texts_batch,
            batch_size=len(texts_batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        owner_batch = owners_np[start : start + batch_size]
        np.add.at(sums, owner_batch, vecs)
        np.add.at(counts, owner_batch, 1)

    # 3) Mean‑pool + L2‑normalise
    doc_embs = sums / counts[:, None]
    doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12
    return doc_embs.astype("float32")


def deduplicate(df: pd.DataFrame, text_col: str, doc_embs: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Deduplicate documents based on embedding similarity.

    Optimized with:
    - Batched FAISS search for large datasets
    - Tqdm progress bars
    - Early filtering of duplicates
    - Optimized connected components calculation
    """
    n = len(doc_embs)
    dim = doc_embs.shape[1]

    # Build FAISS index with progress reporting
    logging.info(f"Building FAISS index for {n} vectors...")
    index = faiss.IndexFlatIP(dim)
    batch_size = min(10000, n)  # Process in batches to show progress

    for i in tqdm(range(0, n, batch_size), desc="Building index"):
        end = min(i + batch_size, n)
        index.add(doc_embs[i:end]) #type: ignore

    logging.info(f"Index built: {index.ntotal} vectors of dimension {dim}")

    # Search for nearest neighbors in batches with progress reporting
    logging.info("Searching for similar pairs...")
    pairs = []
    total_pairs = 0

    # Process in batches to show progress and reduce memory usage
    for i in tqdm(range(0, n, batch_size), desc="Finding duplicates"): # severely slow on 1mln+ datasets - implement single dedup pipeline instead
        end = min(i + batch_size, n)
        batch_size_actual = end - i

        # Get top 2 neighbors (includes self)
        D, I = index.search(doc_embs[i:end], 2) #type: ignore

        # Extract similarities and neighbor indices
        nbrs, sims = I[:, 1], D[:, 1]

        # Create array of original indices for this batch
        orig_indices = np.arange(i, end)

        # Filter duplicates: similarity >= threshold and not self-reference
        mask = (sims >= threshold) & (nbrs != orig_indices)

        if np.any(mask):
            # Create pairs of duplicates [original_idx, neighbor_idx]
            batch_pairs = np.column_stack((orig_indices[mask], nbrs[mask].astype(np.int32)))
            pairs.append(batch_pairs)
            total_pairs += len(batch_pairs)

    # Combine all pairs
    if total_pairs > 0:
        all_pairs = np.vstack(pairs) if pairs else np.empty((0, 2), dtype=np.int32)
        logging.info(f"Found {total_pairs} duplicate pairs")

        # Build sparse graph for connected components
        logging.info("Building connected components graph...")
        row, col = all_pairs.T
        data = np.ones(len(all_pairs), dtype=np.uint8)
        graph = coo_matrix((data, (row, col)), shape=(n, n))

        # Find connected components
        with tqdm(total=1, desc="Clustering duplicates") as pbar:
            n_components, labels = connected_components(graph, directed=False)
            pbar.update(1)

        logging.info(f"Found {n_components} components")

        # Choose representative (longest text) per component with progress reporting
        logging.info("Selecting representative documents...")
        lengths = df[text_col].str.len().to_numpy()
        reps = np.full(n_components, -1, dtype=np.int32)

        # Process in chunks to show progress
        chunk_size = max(1, n // 100)  # Update progress ~100 times
        for chunk_start in tqdm(range(0, n, chunk_size), desc="Selecting representatives"):
            chunk_end = min(chunk_start + chunk_size, n)
            for idx in range(chunk_start, chunk_end):
                comp = labels[idx]
                if reps[comp] == -1 or lengths[idx] > lengths[reps[comp]]:
                    reps[comp] = idx

        keep_idx = np.sort(reps)
    else:
        logging.info("No duplicates found, keeping all documents")
        keep_idx = np.arange(n)

    logging.info(f"Keeping {len(keep_idx)} / {len(df)} rows ({(len(keep_idx)/len(df))*100:.1f}%)")
    return df.iloc[keep_idx].reset_index(drop=True)


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input)
    df = pd.read_parquet(input_path)

    if args.column not in df.columns:
        raise KeyError(f"Column '{args.column}' not found in {args.input}.")

    logging.info(f"Loaded data: {df.shape}; Loading Sentence‑Transformer model...")
    model = SentenceTransformer(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dim = model.get_sentence_embedding_dimension()

    logging.info(f"Loaded model: {args.model}. Embedding documents...")
    doc_embs = embed_all_docs(
        df,
        args.column,
        tokenizer,
        model,
        args.chunk_tokens,
        args.stride,
        args.batch_chunks,
    )
    logging.info("Built embeddings; Clustering duplicates via FAISS...")

    df_dedup = deduplicate(df, args.column, doc_embs, args.threshold)
    kept, total = len(df_dedup), len(df)
    logging.info(f"Semantic dedup kept {kept} / {total} rows (Δ = {total - kept}).")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dedup.to_parquet(output_path, index=True)
    # write_dataset(args.outfile, df_dedup)

    logging.info(f"Written deduplicated data to {output_path}")
    
    
