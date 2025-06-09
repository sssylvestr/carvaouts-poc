import hashlib
import logging
import re
from joblib import Parallel, delayed
from typing import Iterable, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import ahocorasick
from datasketch import MinHash, MinHashLSH
from simhash import Simhash, SimhashIndex
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

logging.getLogger("simhash").setLevel(logging.ERROR)

NON_WORD_RE = re.compile(r"\W+")
RE_WS = re.compile(r"\s+")
TRANS_TABLE = str.maketrans({i: " " for i in range(256) if not chr(i).isalnum()})


def dedup_col(df: pd.DataFrame, text_col="segment") -> pd.DataFrame:
    df[f"{text_col}_norm"] = df[text_col].str.lower().str.replace(RE_WS, " ", regex=True).str.strip() # type: ignore
    df[f"{text_col}_hash"] = df[f"{text_col}_norm"].map(lambda t: hashlib.md5(t.encode()).hexdigest())

    logging.info(
        f"Hash deduplication stats for column '{text_col}': "
        f"{df[f'{text_col}_hash'].nunique()} unique hashes, "
        f"{df[text_col].nunique()} unique texts, "
        f"compression ratio {df[text_col].nunique() / df[f'{text_col}_hash'].nunique():.2f}"
    )
    df.drop_duplicates(subset=[f"{text_col}_hash"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def shingles(text: str, n: int = 8) -> Iterable[str]:
    """Yield *n*-word shingles using a one‑shot regex replace (generator)."""
    tokens = re.sub(r"\W+", " ", text).lower().split()
    for i in range(max(0, len(tokens) - n + 1)):
        yield " ".join(tokens[i : i + n])


def minhash_dedup(
    df: pd.DataFrame,
    col: str = "segment_norm",
    num_perm: int = 64,
    jaccard_thresh: float = 0.9,
    n_jobs: int = -1,
) -> Dict[int, List[int]]:
    """Cluster near‑identical texts using MinHash‑LSH (fast, parallel)."""

    df = df.reset_index(drop=True)
    n = len(df)

    # 3.1  Parallel signature build
    def _build(idx: int, txt: str):
        mh = MinHash(num_perm=num_perm)
        for sh in shingles(txt, 8):
            mh.update(sh.encode())
        return idx, mh

    mhs: List[MinHash] = [None] * n  # type: ignore
    with tqdm_joblib(tqdm(total=n, unit="docs", desc="MinHash build")):
        for idx, mh in Parallel(n_jobs=n_jobs)(delayed(_build)(i, t) for i, t in enumerate(df[col])): # type: ignore
            mhs[idx] = mh

    lsh = MinHashLSH(threshold=jaccard_thresh, num_perm=num_perm)
    keys = [str(i) for i in range(n)]

    for k, mh in zip(keys, mhs):
        lsh.insert(k, mh)

    rows: List[int] = []
    cols: List[int] = []
    for idx, mh in tqdm(enumerate(mhs), total=n, unit="docs", desc="LSH edges"):
        for cand in lsh.query(mh):
            j = int(cand) # type: ignore
            if j <= idx:
                continue
            rows.append(idx)
            cols.append(j)

    if rows:
        data = np.ones(len(rows) * 2, dtype=np.bool_)
        r = np.array(rows + cols, dtype=np.int32)
        c = np.array(cols + rows, dtype=np.int32)
        adj = coo_matrix((data, (r, c)), shape=(n, n), dtype=np.bool_)
        _, labels = connected_components(adj, directed=False, connection="weak")
    else:
        labels = np.arange(n, dtype=np.int32)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx) # type: ignore
    return clusters


def containment_dedup(
    df: pd.DataFrame,
    candidates: Iterable[int],
    col: str = "segment_norm",
    n_jobs: int = -1,
    batch_size: int = 500,  # Smaller batch size to reduce memory usage
    min_length: int = 20,  # Increased min length to avoid excessive matches
    max_automaton_size: int = 100000,  # Max number of patterns in one automaton
) -> Set[int]:
    """Remove texts that are *verbatim substrings* of longer ones.

    Ultra memory-efficient implementation with incremental processing.
    """
    import gc

    # Convert candidates to list to allow multiple iterations
    candidates = list(candidates)
    total_candidates = len(candidates)
    logging.info(f"Starting containment check for {total_candidates} candidates")

    if total_candidates < 2:
        return set(candidates)  # Nothing to compare

    # Use more aggressive memory management for large datasets
    if total_candidates > 100000:
        logging.info("Large dataset detected - using aggressive memory optimization")
        batch_size = min(batch_size, 250)  # Smaller batches
        gc.collect()  # Initial garbage collection

    # Filter out extremely short texts that cause many false positives
    texts = {}
    skipped = 0
    for i in candidates:
        text = df.at[i, col]
        if len(text) >= min_length:
            texts[i] = text
        else:
            skipped += 1

    if skipped > 0:
        logging.info(f"Skipped {skipped} texts shorter than {min_length} characters")

    # Sort by text length (shortest to longest)
    by_len = sorted(texts.items(), key=lambda kv: len(kv[1]))
    keep: Set[int] = set(texts)

    if len(by_len) < 2:
        return keep  # Nothing to compare

    # Approach selection based on dataset size
    if len(by_len) > 500000:
        if n_jobs == -1:
            n_jobs = 16
        logging.warning(f"Dataset too large for Aho-Corasick ({len(by_len)} docs). Switching to sampling approach.")
        return _sampling_containment_check(df, by_len, keep, col, n_jobs, batch_size)

    # Process in small batches to limit memory usage
    # Work with batches of longer texts to reduce memory pressure
    total_batches = (len(by_len) + batch_size - 1) // batch_size
    logging.info(f"Processing in {total_batches} batches with memory optimization")

    # Reverse the list so we start with longest texts
    by_len_reversed = list(reversed(by_len))

    for batch_idx in tqdm(range(0, len(by_len_reversed), batch_size), desc="Batch containment"):
        # Explicit garbage collection at start of each batch
        gc.collect()

        # Get current batch of longer texts
        batch_long = by_len_reversed[batch_idx : batch_idx + batch_size]

        # Find the minimum text length in this batch
        min_batch_len = min(len(long_txt) for _, long_txt in batch_long)

        # Sub-batch the shorter texts to avoid OOM with automaton
        current_keep = [idx for idx in keep if idx not in [long_idx for long_idx, _ in batch_long]]
        shorter_candidates = [(idx, texts[idx]) for idx in current_keep if len(texts[idx]) < min_batch_len]

        if not shorter_candidates:
            continue

        # Process shorter texts in sub-batches to limit automaton size
        for start_pos in range(0, len(shorter_candidates), max_automaton_size):
            # Explicit garbage collection for each sub-batch
            gc.collect()

            end_pos = min(start_pos + max_automaton_size, len(shorter_candidates))
            shorter_texts = shorter_candidates[start_pos:end_pos]

            if not shorter_texts:
                continue

            logging.info(
                f"Processing sub-batch {start_pos//max_automaton_size + 1} with {len(shorter_texts)} patterns"
            )

            # Build automaton only for this sub-batch of shorter texts
            try:
                A = ahocorasick.Automaton()
                for idx, pat in shorter_texts:
                    if idx in keep:  # Double-check we still want this
                        A.add_word(pat, (idx, len(pat)))
                A.make_automaton()
            except Exception as e:
                logging.error(f"Failed to build automaton: {e}")
                # Fall back to alternative for this batch
                continue

            # Check if shorter texts are contained in longer texts of this batch
            discard = set()
            try:
                batch_desc = f"Batch {batch_idx//batch_size + 1}/{total_batches}"
                for idx_long, long_txt in tqdm(batch_long, desc=batch_desc, leave=False):
                    try:
                        for _, (idx_short, _) in A.iter(long_txt):
                            # Only discard if it's a real substring (not the same text)
                            if idx_short != idx_long and idx_short in keep:
                                discard.add(idx_short)
                    except Exception as e:
                        logging.warning(f"Error while processing text {idx_long}: {e}")
                        continue
            except Exception as e:
                logging.error(f"Error in batch processing: {e}")

            # Remove the found substrings
            keep.difference_update(discard)

            # Force garbage collection to free memory from the automaton
            del A
            gc.collect()

    return keep


def _optimized_batch_containment_check(df, by_len, keep, col, n_jobs=16, batch_size=100):
    """Fallback method with optimized batch checking."""
    import gc
    from joblib import Parallel, delayed

    logging.info("Using optimized fallback for containment check")

    if len(by_len) > 100000:
        batch_size = min(batch_size, 200)
        n_jobs = max(1, min(n_jobs, 16))  # Limit parallel jobs to reduce memory

    for batch_idx in tqdm(range(0, len(by_len), batch_size), desc="Fallback containment"):
        gc.collect()

        batch = by_len[batch_idx : batch_idx + batch_size]
        if not batch:
            continue

        def _batch_probe(idx_short, short_txt):
            if idx_short not in keep:
                return idx_short, True

            longer_texts = [
                (idx_long, long_txt)
                for idx_long, long_txt in by_len
                if len(long_txt) > len(short_txt) and idx_long != idx_short and idx_long in keep
            ]

            sub_batch_size = 100
            for sub_start in range(0, len(longer_texts), sub_batch_size):
                sub_batch = longer_texts[sub_start : sub_start + sub_batch_size]
                for idx_long, long_txt in sub_batch:
                    if short_txt in long_txt:
                        return idx_short, False
            return idx_short, True

        try:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_batch_probe)(idx, txt) for idx, txt in batch)
            for idx_short, should_keep in results: # type: ignore
                if not should_keep and idx_short in keep:
                    keep.remove(idx_short)
        except Exception as e:
            logging.error(f"Error in parallel processing: {e}")

        gc.collect()

    return keep


def _sampling_containment_check(df, by_len, keep, col, n_jobs, batch_size):
    """For extremely large datasets, use sampling approach to avoid memory issues."""
    import random

    logging.warning("Using sampling approach for very large dataset")
    sample_size = min(100000, len(by_len) // 2)

    # Take a random sample of longer texts
    longer_half_idx = len(by_len) // 2
    sampled_short = random.sample(by_len[:longer_half_idx], min(sample_size // 2, longer_half_idx))
    sampled_long = random.sample(by_len[longer_half_idx:], min(sample_size // 2, len(by_len) - longer_half_idx))

    sampled_by_len = sorted(sampled_short + sampled_long, key=lambda kv: len(kv[1]))
    sampled_keep = {idx for idx, _ in sampled_by_len}

    # Process the sample with existing logic
    result_keep = _optimized_batch_containment_check(df, sampled_by_len, sampled_keep, col, n_jobs, batch_size)

    # Filter original keep set based on sampling results
    final_keep = {idx for idx in keep if idx not in sampled_keep or idx in result_keep}

    logging.info(f"Sampling approach kept {len(final_keep)}/{len(keep)} entries")
    return final_keep


def simhash_dedup(
    df: pd.DataFrame,
    candidates: Iterable[int],
    col: str = "segment_norm",
    ngram: int = 4,
    hamming_thresh: int = 3,
    show_progress: bool = True,
) -> Set[int]:
    """Near‑duplicate removal via 64‑bit SimHash (sequential, low‑RAM & safe)."""

    def _tok4(text: str) -> List[str]:
        toks = text.lower().translate(TRANS_TABLE).split()
        if len(toks) < ngram:
            return toks
        return [" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)]

    keeper: Set[int] = set()
    index = SimhashIndex([], k=hamming_thresh)

    iterator = tqdm(candidates, desc="SimHash stream", unit="docs", disable=not show_progress)
    for idx_int in iterator:
        sim = Simhash(_tok4(df.at[idx_int, col]), f=64)
        if index.get_near_dups(sim):
            continue
        index.add(str(idx_int), sim)
        keeper.add(idx_int)
    return keeper


def deduplicate(df: pd.DataFrame, col: str = "segment_norm") -> pd.DataFrame:
    """Full three‑stage dedup; returns a *new* DataFrame of unique docs."""

    print("[1/3] MinHash clustering …")
    clusters = minhash_dedup(df, col)
    canon_indices = {max(idxs, key=lambda i: len(df.at[i, col])) for idxs in clusters.values()}

    print("[2/3] Skipping substring containment …")
    # print("[2/3] Substring containment filter …")
    # canon_indices = containment_dedup(df, canon_indices, col)

    print("[3/3] SimHash fuzzy filter …")
    canon_indices = simhash_dedup(df, canon_indices, col)

    return df.loc[sorted(canon_indices)].reset_index(drop=True)
