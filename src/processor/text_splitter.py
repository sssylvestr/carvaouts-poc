import os, logging

from functools import partial, lru_cache
from typing import List, Tuple, Sequence, Dict

import torch
import pandas as pd

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from src.processor.text_tiling_tokenizer import CustomTextTilingTokenizer

MAX_LEN = 512  # max BPE length for a single tile
PARAM_KEYS = ("w", "k", "smoothing_width", "smoothing_rounds", "cutoff_policy")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@lru_cache(maxsize=24)
def _get_splitter(params_tuple):
    params = dict(zip(PARAM_KEYS, params_tuple))
    return CustomTextTilingTokenizer(**params)

def _tiles_depth_static(text: str, params_tuple) -> List[Tuple[str, float, None]]:
    splitter = _get_splitter(params_tuple)
    try:
        tiles = splitter.tokenize(text)
    except ValueError: # up to 1 %, mostly too short texts
        return [(text, 0., None)]
    depth = [None] + splitter.depth_scores[: len(tiles) - 1]
    return [(t, d, None) for t, d in zip(tiles, depth)] # type: ignore

def _first_last_static(seg_list: Sequence[Tuple[str, float, None]]):
    out = []
    for idx, (tile, _, _) in enumerate(seg_list):
        sents = sent_tokenize(tile)
        if not sents:
            continue
        fst = sents[0]
        lst = sents[-1] if len(sents) >= 2 else sents[0]  # duplicate
        out.append((idx, fst, lst))
    return out


def rechunk(tile: str, tokenizer, max_len: int = MAX_LEN) -> List[str]:
    """
    Split an oversize tile so every chunk ≤ max_len BPE tokens.
    Tokenises each sentence *once*; O(N) time and memory.
    """
    sents = sent_tokenize(tile)
    if not sents:
        return []

    lens  = [len(ids) for ids in tokenizer(sents,add_special_tokens=False)["input_ids"]]

    if sum(lens) <= max_len:
        return [tile]

    out, buf, acc = [], [], 0
    for sent, sent_len in zip(sents, lens):
        if sent_len > max_len:
            if buf:
                out.append(" ".join(buf)); buf, acc = [], 0

            toks = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(toks), max_len):
                chunk_ids = toks[i : i + max_len]
                out.append(tokenizer.decode(chunk_ids))
            continue

        if acc + sent_len > max_len:
            out.append(" ".join(buf))
            buf, acc = [], 0
        buf.append(sent); acc += sent_len

    if buf:
        out.append(" ".join(buf))
    return out



class ParallelTextTiler:
    def __init__(
        self,
        batch_size: int = 512,
        calc_stats: bool = False,       
        num_workers: int = os.cpu_count() - 2, #type: ignore
        sbert_model_id: str = "intfloat/multilingual-e5-small",
        text_tiling_params: Dict | None = None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  
        self.batch_size = batch_size
        self.calc_stats = calc_stats
        
        self.text_tiling_params = text_tiling_params or dict(
            w=25, k=6, smoothing_width=4, smoothing_rounds=2, cutoff_policy="HC")       
        
        if self.calc_stats:
            logging.info("loading SentenceTransformer model for edge scores")         
            if DEVICE=='cuda': #JIT warm-up
                self.model = SentenceTransformer(sbert_model_id, device=DEVICE).half()

                _ = self.model.encode(["warm-up"] * self.batch_size, 
                        batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
            else:
                self.model = SentenceTransformer(sbert_model_id, device='cpu')    
                torch.set_num_threads(min(8, num_workers))
            self._tokenizer = self.model.tokenizer
        else:
            logging.info(f"Loading tokenizer only: {sbert_model_id}")        
            self._tokenizer = AutoTokenizer.from_pretrained(sbert_model_id, use_fast=True)

    def segment_dataframe(self, df: pd.DataFrame, col: str) -> pd.Series:
        params_tuple = tuple(self.text_tiling_params[k] for k in PARAM_KEYS)
        logging.info(f"Step 1: Tiling texts")

        tiles_func = partial(_tiles_depth_static, params_tuple=params_tuple)
        segments   = df[col].parallel_apply(tiles_func)
        logging.info(f"Segmented {len(segments)} articles into tiles; Re-chunking longer tiles")
        
               
        if not self.calc_stats:
            return segments.apply(lambda lst: [t for t, _,_ in lst])

        # segments = segments.apply( # rechunking for metrics calculation
        #         lambda segs: [# actually breaks the semantics, but we need to calculate pseudo-metrics
        #             (t2, d, e)
        #             for t, d, e in segs
        #             for t2 in rechunk(t, self._tokenizer) # very slow; parallelize if decided to use
        #         ]
        #     )
        
        logging.info(f"Step 2: Extracting first/last sentences for pseudo-metrics")
        pairs_per_doc = segments.parallel_apply(_first_last_static)

        pairs = [
            (doc, idx, fst, lst)
            for doc, lsts in pairs_per_doc.items()
            for idx, fst, lst in lsts
        ]
        if not pairs:
            return segments

        flat = [s for p in pairs for s in (p[2], p[3])]
        logging.info(f"Step 3: Encoding {len(flat)} first/last sentences in pairs")
        
        cosines: List[float] = [] ## cacl edge scores between first and last sentences
        for i in tqdm(range(0, len(flat), self.batch_size), desc="SBERT edge pass"):
            with torch.inference_mode():
                embs = self.model.encode(flat[i:i+self.batch_size], 
                                    batch_size=self.batch_size,
                                    convert_to_tensor=True,
                                    normalize_embeddings=False,
                                    show_progress_bar=False)
                cos   = util.cos_sim(embs[0::2], embs[1::2]).diagonal().tolist()
                cosines.extend(cos)

        for (doc, idx, *_), edge in zip(pairs, cosines):
            tile, depth, _ = segments.at[doc][idx]
            segments.at[doc][idx] = (tile, depth, edge)

        return segments


def tile_texts(
    df: pd.DataFrame,
    col: str = "body_cleaned",
    wc_col: str = "word_count"
) -> pd.DataFrame:
    """
    Segment each article into topical tiles in parallel.
    Parameters
    ----------
    df : DataFrame
        Must contain `col` and `wc_col`.

    Returns
    -------
    DataFrame (exploded): one row per segment.
    """
    if col not in df.columns or wc_col not in df.columns:
        raise ValueError(f"Columns '{col}' and '{wc_col}' must be present.")

    df = df.copy()
    
    tiler = ParallelTextTiler()
    
    df["segments"] = tiler.segment_dataframe(df, col=col) # without stats
    df["num_segments"] = df["segments"].apply(len)
    
    df = df.explode("segments", ignore_index=True)
    df = df.rename(columns={"segments": "segment"})
    
    df['segment_id'] = range(len(df)) # separate tile id for each segment - imp when saving parquet by month
    
    return df



