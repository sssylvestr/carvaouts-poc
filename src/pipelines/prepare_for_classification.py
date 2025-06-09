import argparse
import logging
import os
import pandas as pd

from tqdm import tqdm

from src.mapping.company_mapper import map_companies
from src.utils.regions import filter_by_region

tqdm.pandas()
logging.basicConfig(level=logging.INFO)


cols2use = ['source_name','title','company_codes','modification_date','region_codes','segment','num_segments','an', 'segment_id'] #post-processing and filtering will be later

## careful with csv/parquet
def parse_args():
     parser = argparse.ArgumentParser(description="Merging deduplicated data with original dataset; additional filtering by region")

     parser.add_argument('--input_dataset','-i', required=True, help="Merge deduplicated data with original dataset")
     parser.add_argument('--deduplicated_dataset','-d',required=True, help="Path to the file with deduplicated article segments")
     parser.add_argument('--output_file','-o', required=True, help="Path to save the merged output file; should be .csv")

     return parser.parse_args()


if __name__ == "__main__":
     args = parse_args()
     logging.info("Starting merging deduplication results...")

     # Load the original dataset
     original_df = pd.read_parquet(args.input_dataset, columns=cols2use)
     original_df = filter_by_region(original_df)
     
     # Load the deduplicated dataset
     dedup_df = pd.read_parquet(args.deduplicated_dataset)

     # Save the merged dataset to the specified output file
     logging.info(f"Original dataset loaded with shape: {original_df.shape}")
     df2classify = original_df[original_df.segment_id.isin(dedup_df.segment_id) & original_df.regions_relevant].copy()
     df2classify = df2classify.loc[ :, ~df2classify.columns.isin(['region_codes_list','regions_relevant'])]
     logging.info(f"Filtered dataset for classification with shape: {df2classify.shape}")
     # additional mappings
     logging.info("Mapping company codes to names...")
     df2classify = map_companies(df2classify, new_col="company_names")

     df2classify.to_csv(args.output_file, index=False)

     logging.info(f"Merged dataset saved to {args.output_file}")