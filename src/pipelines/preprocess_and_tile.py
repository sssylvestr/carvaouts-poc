import argparse
import logging
import os
import pandas as pd
import time
from pandarallel import pandarallel


from src.processor.preprocessor import preprocess_extracted_data
from src.processor.text_splitter import tile_texts
from src.utils.helpers import write_dataset


logging.basicConfig(level=logging.INFO)
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count() -2) #type: ignore

if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Fast TextTiling in parallel")
    argp.add_argument("--infile","-i",type=str, required=True, help="csv with 'body' and 'word_count'")
    argp.add_argument("--outdir","-o", type=str,required=True, help="Destination Parquet (exploded)")
    argp.add_argument("--test_run", type=bool, help="Will do a test run with 1000 rows", default=False)
    argp.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = argp.parse_args()

    logging.info("Starting data preprocessing...\n")
    t0 = time.perf_counter()

    df = pd.read_csv(args.infile, low_memory=False)
    if args.test_run:
        df = df.sample(1000, random_state=42).reset_index(drop=True)
    df = preprocess_extracted_data(df)

    logging.info(f"Preprocessed data: {df.shape}\n")
    logging.info("Starting text segmentation...\n")

    df = tile_texts(df[:].copy())
    dur = time.perf_counter() - t0
    logging.info(f"Segmented data: {df.shape}; Writing dataset to {args.outdir}\n")
    write_dataset(args.outdir, df)
    logging.info(f"Done in {dur:0.1f}s → {args.outdir}")
