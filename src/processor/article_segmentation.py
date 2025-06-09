#!/usr/bin/env python3
"""
Article Segmentation Script

This script processes articles from a CSV file, segments them using TextTilingTokenizer,
and outputs a CSV file with the segmented articles.

Requirements:
    - pandas
    - tqdm-joblib
    - nltk
    - bs4 (BeautifulSoup)
    - joblib
"""

import re
import os
import argparse
from functools import partial
from multiprocessing import Value

import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TextTilingTokenizer
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm.autonotebook import tqdm

# Regular expression for tokenizing text
TOKEN_RX = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def factiva_wordcount(html: str) -> int:
    """
    Approximate Factiva 'WC' very closely (±0–2 % on real wires).
    1. Strip markup quickly with lxml‑backed BeautifulSoup.
    2. Replace hyphens & slashes with space so they split tokens.
    3. Count regex‑defined tokens.
    
    Args:
        html (str): The HTML content to count words from
        
    Returns:
        int: The word count
    """
    # -------- strip HTML completely --------
    txt = BeautifulSoup(html, "lxml").get_text(" ")

    # -------- normalise separators ---------
    txt = txt.replace("‑", "-")           # unicode hyphen
    txt = re.sub(r"[/-]", " ", txt)       # split hyphen/slash compounds
    txt = txt.replace("\u00A0", " ")      # non‑breaking sp

    # -------- tokenise & count ------------
    return len(TOKEN_RX.findall(txt))


def _tile(doc, wc, wc_min=100):
    """
    Segment a document using TextTilingTokenizer if it exceeds minimum word count.
    
    Args:
        doc (str): Document text to segment
        wc (int): Word count of the document
        wc_min (int): Minimum word count for segmentation
        
    Returns:
        tuple: ([segments], error_flag)
    """
    if wc < wc_min:
        return [doc], 0           # segments, errflag

    try:
        return tokenizer.tokenize(doc), 0
    except Exception as e:
        return [doc], 1


def segment_articles(input_file, output_file=None, body_col='body', word_count_col='word_count'):
    """
    Segment articles from a CSV file using TextTilingTokenizer.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output CSV file. If None, will use input_file with '_segmented' suffix
        body_col (str): Column name for article body
        word_count_col (str): Column name for word count
        
    Returns:
        pd.DataFrame: DataFrame with segmented articles
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Configure tqdm for pandas
    tqdm.pandas()
    
    # Get article bodies and word counts
    bodies = df[body_col].tolist()
    word_counts = df[word_count_col].tolist()
    
    # Set up parallel processing
    CPUS = os.cpu_count()
    ERRORS = Value("i", 0)  # shared int (32-bit)
    
    # Initialize tokenizer
    global tokenizer
    tokenizer = TextTilingTokenizer()
    
    print("Segmenting articles...")
    with tqdm_joblib(tqdm(total=len(bodies), desc="Tiling")):
        results = Parallel(n_jobs=CPUS, backend="multiprocessing", batch_size=4)(
            delayed(_tile)(doc, wc) for doc, wc in zip(bodies, word_counts)
        )
    
    # Unpack results
    segments, err_flags = zip(*results)
    
    # Add segments to DataFrame
    df["segments"] = segments
    df['num_segments'] = df['segments'].apply(lambda x: len(x))
    error_count = sum(err_flags)
    
    print(f"{error_count} documents raised an error during tiling.")
    
    # Explode segments to create one row per segment
    df_exploded = df.explode('segments')
    
    # Save results if output file specified
    if output_file is None:
        # Create default output filename
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_segmented{ext}"
    
    if output_file:
        print(f"Saving segmented articles to {output_file}...")
        df_exploded.to_csv(output_file, index=False)
    
    return df_exploded


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Segment articles using TextTilingTokenizer')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output CSV file')
    parser.add_argument('--body-col', default='body', help='Column name for article body (default: body)')
    parser.add_argument('--word-count-col', default='word_count', help='Column name for word count (default: word_count)')
    
    args = parser.parse_args()
    
    df_exploded = segment_articles(
        args.input, 
        args.output,
        args.body_col,
        args.word_count_col
    )
    
    print(f"Processing complete. Segmented {len(df_exploded)} segments from {len(df_exploded['num_segments'].unique())} articles.")


if __name__ == "__main__":
    main()
