import argparse
import logging
import pandas as pd
import time

from src.processor.hash_and_dedup import deduplicate, dedup_col
from src.utils.helpers import write_dataset, read_dataset

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    argp = argparse.ArgumentParser(description="Hash and deduplicate articles")
    argp.add_argument("--infile", "-i", required=True, help="parquet (or dir) with 'body' and 'word_count'")
    argp.add_argument("--outfile", "-o", required=True, help="Destination Parquet (deduplicated); note that this currently is a single file, not a directory")
    argp.add_argument("--col", default="segment", help="Column to deduplicate (default: segment)")

    args = argp.parse_args()

    logging.info("Starting deduplication of the data..")
    t0 = time.perf_counter()

    # Use string_cols parameter to control memory usage
    logging.info(f"Reading column '{args.col}' to Python strings")
    # df = read_dataset(args.infile, batch_rows=100_000, string_cols=args.col)
    df = pd.read_parquet(args.infile, columns=["segment","segment_id"])
    logging.info(f"Loaded data: {df.shape}")
    df = dedup_col(df, text_col=args.col)
    logging.info(f"Shape after straight deduplication: {df.shape}")
    df = deduplicate(df, col=args.col)

    dur = time.perf_counter() - t0
    logging.info(f"Done in {dur:0.1f}s → {args.outfile}")
    logging.info(f"Deduplicated data: {df.shape}")
    # write_dataset(args.outfile, df)
    df.to_parquet(args.outfile, index=True)
