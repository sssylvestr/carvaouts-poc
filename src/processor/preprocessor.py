import re
import logging

from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import sent_tokenize

import pandas as pd

from typing import Collection, Iterable, List
from tqdm import tqdm

logger = logging.getLogger(__name__)
HTML_TAG_SNIF = re.compile(r"<(p|div|br|h[1-6]|table|ul|ol)\b", re.I)


def most_common_lines(
    texts: Collection[str],
    top_k: int = 1000,
    min_pct_docs: float = 0.03,
    min_len: int = 10,
    max_len: int = 160,
) -> pd.DataFrame:
    """
    Return the `top_k` most frequent full lines across the corpus,
    excluding very short or very long ones (likely not boiler-plate).
    """
    n_texts = len(texts)
    counter = Counter()
    for doc in tqdm(texts):
        for line in doc.splitlines():
            stripped = line.strip()
            if min_len <= len(stripped) <= max_len:
                counter[stripped] += 1

    df = pd.DataFrame(counter.most_common(top_k), columns=["line", "count"])
    df["pct_docs"] = df["count"] / n_texts
    df = df.query("pct_docs >= @min_pct_docs").sort_values("count", ascending=False)
    return df

def html_to_parbreaks(html: str) -> str:
    """Very small helper: replace block tags with '\n\n', strip the rest."""
    soup = BeautifulSoup(html, "lxml")
    for br in soup.find_all("br"):
        br.replace_with("\n") # type: ignore
    for tag in soup.find_all(True):
        if tag.name.lower() in {"p", "div", "li", "table", "tr",  # type: ignore
                                "ul", "ol", "section", "header",
                                "footer", "h1", "h2", "h3", "h4", "h5", "h6"}:
            tag.insert_before("\n\n")
            tag.insert_after("\n\n")
    text = soup.get_text(" ", strip=False)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def compile_boilerplate_regex(lines: Iterable[str]) -> re.Pattern:
    """
    Build a single regex OR-ing escaped boiler-plate lines.
    Use NON-capturing group `(?:...)` so re.sub can drop them easily.
    """
    escaped = [re.escape(l) for l in lines]
    pattern = r"^(?:%s)$" % "|".join(escaped)
    return re.compile(pattern, flags=re.MULTILINE)

def inject_paragraph_breaks(text: str,min_tokens: int = 40, max_tokens: int = 80) -> str:
     """
     Insert '\n\n' paragraph markers so that each paragraph has roughly
     min_tokens–max_tokens whitespace words.

     Needed to make TextTiler work with texts without paragraph breaks.
     """
     
     sentences = sent_tokenize(text)
     out, buf, n = [], [], 0
     
     for sent in sentences:
          tok = len(sent.split())
          buf.append(sent)
          n += tok
          
          if n >= max_tokens or (n>=min_tokens and re.search(r"[.!?]$", sent)):
               out.append(" ".join(buf))
               buf, n = [], 0
     if buf:
          out.append(" ".join(buf))
               
     return "\n\n".join(out)

def clean_document(
    raw_text: str,
    boilerplate_re: re.Pattern,
    min_caps_ratio: float = 0.6,
    max_short_tokens: int = 5,
) -> str:
    """
    1) Strip boiler-plate lines;
    2) Normalise paragraph breaks to exactly one *blank* line between paragraphs;
    3) Merge tiny paragraphs (datelines, bullets) into the next paragraph.
    """
    text = boilerplate_re.sub("", raw_text)
    text = re.sub(r"\r\n|\r", "\n", text) 
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    parts = text.split("\n\n")
    good_parts = []
    buffer = []
    for p in parts:
        tokens = p.split()
        caps_ratio = sum(t.isupper() for t in tokens) / max(len(tokens), 1)
        if (len(tokens) <= max_short_tokens) or (caps_ratio >= min_caps_ratio):
            buffer.append(p)
            continue

        if buffer:
            p = " ".join(buffer) + " " + p
            buffer = []
        good_parts.append(p)
    if buffer:                          # dangling buffer at EOF
        good_parts.append(" ".join(buffer))

    return "\n\n".join(good_parts)

def preprocess_document(
    raw_text: str,
    boilerplate_re: re.Pattern,
    min_caps_ratio: float = 0.6,
    max_short_tokens: int = 5,
    inj_min_tokens: int = 40,
    inj_max_tokens: int = 80,
    trigger_ratio: float = 1.5,
) -> str:
    """
    1) Run `clean_document`.
    2) If *any* paragraph still exceeds `trigger_ratio * inj_max_tokens`
       OR there is only one paragraph at all, call `inject_paragraph_breaks`.
    Returns the fully pre-processed text, ready for TextTiling.
    """
    looks_like_html = HTML_TAG_SNIF.search(raw_text)
    
    text = html_to_parbreaks(raw_text) if looks_like_html else raw_text    
    text = clean_document(
        text,
        boilerplate_re,
        min_caps_ratio=min_caps_ratio,
        max_short_tokens=max_short_tokens,
    )
    
    paragraphs = text.split("\n\n")
    if len(paragraphs) == 1:
        need_injection = True
    else:
        longest = max(len(p.split()) for p in paragraphs)
        need_injection = longest > trigger_ratio * inj_max_tokens

    if need_injection:
        text = inject_paragraph_breaks(
            text,
            min_tokens=inj_min_tokens,
            max_tokens=inj_max_tokens,
        )
    return text

def preprocess_extracted_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies initial preprocessing steps to the raw extraction DataFrame.

    Args:
        df: Raw DataFrame from Factiva extraction

    Returns:
        Preprocessed DataFrame with cleaned and standardized data
    """
    logger.info(f"Starting preprocessing on DataFrame with shape {df.shape}")

    # Drop irrelevant columns (adjust list as needed)
    cols_to_drop: List[str] = [
        "dateline",
        "currency_codes",
        "availability_datetime",
        "copyright",
        "language_code",
        "region_of_origin",
        "byline",
        "person_codes",
        "publication_date",
        "publisher_name",
        "credit",
        "art",
        "document_type",
        "modification_datetime",
        "market_index_codes",
    ] + [c for c in df.columns if c.endswith("_exchange")]

    df_processed = df.drop(columns=cols_to_drop, errors="raise")
    logger.info(f"Dropped columns: {cols_to_drop}. New shape: {df_processed.shape}")

    # Convert timestamp columns
    dt_columns: List[str] = [
        "modification_date",
        "modification_datetime",
        "ingestion_datetime",
        "publication_datetime",
        "publication_date",
    ]
    for col in dt_columns:
        if col in df_processed.columns:
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], unit="ms", errors="raise")
                logger.debug(f"Converted column '{col}' to datetime.")
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to datetime: {e}")
        else:
            logger.debug(f"Timestamp column '{col}' not found, skipping conversion.")

    # Aggregate company codes
    cc_columns = sorted([c for c in df_processed.columns if c.startswith("company_codes")])
    if cc_columns:
        logger.info(f"Aggregating company code columns: {cc_columns}")
        for col in cc_columns:
            # Clean up comma-separated strings, remove duplicates and empty strings
            df_processed[col] = df_processed[col].apply(
                lambda x: (
                    ",".join(sorted(list(set(str(x).lstrip(",").split(",")) - {""})))
                    if pd.notna(x) and isinstance(x, str)
                    else ""
                )
            )  # check very carefully
            df_processed[col] = df_processed[col].fillna("")

        df_processed["all_company_codes"] = df_processed[cc_columns].agg(
            lambda x: ",".join(sorted(list(set(",".join(x).split(",")) - {""}))), axis=1
        )
        df_processed = df_processed.drop(columns=cc_columns, errors="raise")
        df_processed = df_processed.rename(columns={"all_company_codes": "company_codes"})
        logger.info("Aggregated company codes into 'company_codes' column.")
    else:
        logger.info("No columns starting with 'company_codes' found for aggregation.")

    # Define and reorder final columns (adjust as needed)
    final_columns_order: List[str] = [
        "source_name",
        "title",
        "snippet",
        "body",
        "section",
        "word_count",
        "source_code",
        "industry_codes",
        "company_codes",
        "subject_codes",
        "publication_datetime",
        "ingestion_datetime",
        "modification_date",
        "region_codes",
        "an",
        "action",
    ]
    assert all(
        [c in df.columns for c in final_columns_order]
    ), f"Missing columns: {[c for c in final_columns_order if c not in df.columns]}"

    df_processed = df_processed[final_columns_order]

    df_processed["pub_date"] = pd.to_datetime(df_processed["publication_datetime"])
    df_processed["year"] = df_processed["pub_date"].dt.year
    df_processed["month"] = df_processed["pub_date"].dt.month

    logger.info(f"Reordered columns. Final shape: {df_processed.shape}")
    logger.info(f"Cleaning dataset")
    
    boilerplate_lines = most_common_lines(df['body'])    
    bp_regex = compile_boilerplate_regex(boilerplate_lines['line'].tolist())
    df_processed.loc[:,'body_cleaned'] = df_processed['body'].parallel_apply(lambda x: preprocess_document(x, bp_regex))
    
    return df_processed
