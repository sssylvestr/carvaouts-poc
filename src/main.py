#!/usr/bin/env python
"""
Carveouts Detection Pipeline

This script orchestrates the full workflow for detecting and analyzing
potential corporate carveouts from Factiva news articles.

Usage:
    python main.py --extract      # Run extraction from Factiva
    python main.py --process      # Process previously extracted data
    python main.py --classify     # Run LLM classification on processed data
    python main.py --export       # Export and format results
    python main.py                # Run full pipeline
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Local imports
from factiva.extraction import run_explain, run_extraction, avro_to_dataframe
from factiva.processors import preprocess_extracted_data, limit_text_length, drop_body_duplicates
from factiva.taxonomy import get_taxonomy_client, get_company_code_mapping, get_industry_code_mapping
from mapping.company_mapper import map_company_codes_to_names
from mapping.industry_mapper import map_industry_codes_to_names
from export.formatters import process_and_export_results
from utils.helpers import chunk_list, safe_read_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"carveouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants and configurations
EXTRACTION_DIR = os.getenv("EXTRACTION_DIR", "../../extractions")
DEFAULT_AVRO_PATH = os.path.join(
    EXTRACTION_DIR, "dj-synhub-extraction-lkbi9fy6zepu8rcjuxqhjwkbld52wgt0-l0fy7lkzhf/part-000000000000.avro"
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "../../demo")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

# Define configurations for the pipeline
CONFIG = {
    "extraction": {
        "query": {
            "query": {
                "where": (
                    "language_code='en' AND publication_datetime >= '2024-01-01 00:00:00' "
                    "AND region_codes LIKE '%eurz%' "
                    "AND (subject_codes LIKE '%cactio%' OR subject_codes LIKE '%cspinoff%' "
                    "OR subject_codes LIKE '%cdivest%' OR subject_codes LIKE '%cmger%' OR subject_codes LIKE '%crestruc%')"
                    "AND (body LIKE '%plan to%' OR body LIKE '%intend to%' OR body LIKE '%looking to%' "
                    "OR body LIKE '%considering%' OR body LIKE '%mulling%' OR body LIKE '%weighing%')"
                    "AND (body LIKE '%divest%' OR body LIKE '%sell%' OR body LIKE '%spin-off%' "
                    "OR body LIKE '%dispose%' OR body LIKE '%carve-out%')"
                ),
                "includesList": {
                    "industry_codes": ["2ce88edb-3f5e-43c5-bf4b-48eb22624ff1"],  # Finance industry list
                },
            }
        },
        "target_root": EXTRACTION_DIR,
    },
    "processing": {
        "text_max_length": 2000,
        "output_csv_path": os.path.join(EXTRACTION_DIR, "preprocessing_result.csv"),
    },
    "classification": {
        "model_name": os.getenv("LLM_MODEL", "o3-mini"),
        "batch_size": BATCH_SIZE,
        "max_concurrent": 5,
        "partial_every": 10,
        "partial_dir": "../../outputs/partial_outputs",
        "final_path": os.path.join(EXTRACTION_DIR, "classification_results.csv"),
        "business_request": (
            """
### Deals search criteria
* Completed date (last 10 years)
* Geography (Europe)
* Deal technique (Divestment)
* Sector (Financial services)
* Size (TBD)
"""
        ),
    },
    "export": {"output_dir": OUTPUT_DIR},
}


class CarveOutAssessment(BaseModel):
    """Schema for the expected output of the LLM assessment."""

    is_co: bool = Field(description="Is the article about a future corporate carve-out?")
    co_confidence: float = Field(description="Confidence level of the carve-out assessment (0-1)")
    target_company_code: str = Field(description="Code of the company that may be open to divest/carve-out")
    is_relevant: bool = Field(description="Is the article relevant to the business request?")
    short_reasoning: str = Field(description="Justification for the answers, not more than 2 sentences")


def setup_factiva_client():
    """Initialize the Factiva taxonomy client and retrieve mappings."""
    user_key = os.getenv("FACTIVA_USER_KEY")
    if not user_key:
        logger.error("FACTIVA_USER_KEY not found in environment variables or .env file")
        sys.exit(1)

    try:
        tax_client = get_taxonomy_client(user_key)
        company_mapping = get_company_code_mapping(tax_client)
        industry_mapping = get_industry_code_mapping(tax_client)
        return tax_client, company_mapping, industry_mapping
    except Exception as e:
        logger.error(f"Failed to set up Factiva client: {e}")
        sys.exit(1)


def extract_data():
    """Extract data from Factiva using the configured query."""
    logger.info("Starting data extraction from Factiva")
    query = CONFIG["extraction"]["query"]
    target_root = CONFIG["extraction"]["target_root"]

    # First, check how many documents would match
    logger.info("Running explain query to check document count")
    try:
        final_status, samples = run_explain(query, num_samples=3)
        if not final_status:
            logger.error("Explain query failed, cannot proceed with extraction")
            return False

        count = final_status["data"]["attributes"].get("counts", "unknown")
        logger.info(f"Explain query shows {count} matching documents")

        # Ask for confirmation if more than 1000 docs
        if isinstance(count, int) and count > 1000:
            confirm = input(f"Query will extract {count} documents. Proceed? [y/N]: ")
            if confirm.lower() != "y":
                logger.info("Extraction cancelled by user")
                return False
    except Exception as e:
        logger.error(f"Error in explain step: {e}")
        return False

    # Run the actual extraction
    try:
        job_id = run_extraction(query, target_root=target_root)
        logger.info(f"Extraction complete, job ID: {job_id}")
        return job_id
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return False


def process_data(avro_path=DEFAULT_AVRO_PATH):
    """
    Process the extracted data, including:
    - Converting Avro to DataFrame
    - Cleaning and preprocessing
    - Mapping codes to human-readable names
    """
    logger.info(f"Processing data from {avro_path}")

    # Get Factiva client and mappings
    _, company_mapping, industry_mapping = setup_factiva_client()

    # 1. Load Avro data
    try:
        df = avro_to_dataframe(avro_path)
        if df.empty:
            logger.error("No data found in Avro file")
            return None
        logger.info(f"Loaded {len(df)} records from Avro file")
    except Exception as e:
        logger.error(f"Error loading Avro data: {e}")
        return None

    # 2. Preprocess data
    try:
        df = preprocess_extracted_data(df)
        logger.info("Data preprocessing complete")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None

    # 3. Map company and industry codes
    try:
        df["company_names"] = map_company_codes_to_names(df["company_codes"], company_mapping)
        df["industries"] = map_industry_codes_to_names(df["industry_codes"], industry_mapping)
        logger.info("Mapped company and industry codes to names")
    except Exception as e:
        logger.error(f"Error mapping codes to names: {e}")
        # Continue anyway, as this is not critical

    # 4. Limit text length and drop duplicates
    try:
        df = limit_text_length(df, column="body", max_length=CONFIG["processing"]["text_max_length"])
        df = drop_body_duplicates(df)
        logger.info(f"Final processed data has {len(df)} records")
    except Exception as e:
        logger.error(f"Error in final processing steps: {e}")
        return None

    # 5. Save processed data
    output_path = CONFIG["processing"]["output_csv_path"]
    try:
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")

    return df


def classify_data(df=None):
    """
    Run LLM classification on the processed data.

    If df is None, will attempt to load from the default processing output path.
    """
    # 1. Import LLM dependencies only when needed
    try:
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from llm_utils.factory import construct_llm
    except ImportError:
        logger.error("Required LLM packages not found. Install with: pip install langchain-core llm_utils")
        return None

    # 2. Load data if not provided
    if df is None:
        input_path = CONFIG["processing"]["output_csv_path"]
        df = safe_read_csv(input_path)
        if df is None:
            logger.error(f"Failed to load processed data from {input_path}")
            return None
        logger.info(f"Loaded {len(df)} records for classification")

    # 3. Prepare input data for LLM
    cols_to_use = ["source_name", "title", "body", "company_names", "company_codes", "modification_date"]
    df_to_classify = df[cols_to_use]

    # 4. Create the LLM prompt
    system_msg = SystemMessagePromptTemplate.from_template(
        "You are an expert in corporate carve-outs. "
        "You are given a news article and a list of companies mentioned in it to understand whether a company is potentially interested in divesting a part of its business. "
        "You are also given a business request that outlines the conditions under which the article is relevant to the business request.\n\n"
        "Your task is to:\n"
        "1. Answer whether the article is about a future corporate carve-out or not.\n"
        "2. If it is about a corporate carve-out, extract the code of the company that is being carved out and the name of the company that may be open to divest.\n"
        "3. Answer whether the article is relevant to the conditions of the business request given to us.\n"
        "Each company mentioned in the article is represented by its code. The codes are unique identifiers for each company.\n\n"
        "Provide justification for your answer."
        "Please note that we are interested only in future carve-outs, not past ones. We are interested in the willingness of the company to divest a part of its business, the fact that it has already done so means it is already late.\n\n"
        "Business request: {business_request}\n"
        "Additional guide: some indicators of a corporate carve-out are: new CEO, management change, strategic shift, legal complications or changes in legislature, IPO, dividends and bonds machuring, focus on core assets, discontinue of operations, M&A activity"
    )

    human_msg = HumanMessagePromptTemplate.from_template(
        "news source: {news_source}\n"
        "article title: {article_title}\n"
        "article body: {article_body}\n"
        "companies: {companies}\n\n"
        "company codes: {company_codes}\n"
        "note that the company codes are unique identifiers for each company and you should use these exact codes in your answer.\n"
        "Outputs:"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

    # 5. Set up the LLM chain
    cfg = CONFIG["classification"]
    business_request = cfg["business_request"]

    def df_to_prompt_inputs(df):
        """Convert DataFrame to list of prompt input dictionaries."""
        df_renamed = df.rename(
            columns={
                "source_name": "news_source",
                "title": "article_title",
                "body": "article_body",
                "company_names": "companies",
                "company_codes": "company_codes",
                "modification_date": "modification_date",
            }
        )
        df_renamed["business_request"] = business_request

        cols = [
            "news_source",
            "article_title",
            "article_body",
            "companies",
            "company_codes",
            "modification_date",
            "business_request",
        ]
        return df_renamed[cols].reset_index().to_dict(orient="records")

    # 6. Create batches of inputs
    prompt_inputs = df_to_prompt_inputs(df_to_classify)
    chunks = chunk_list(prompt_inputs, cfg["batch_size"])
    logger.info(f"Created {len(chunks)} batches of size {cfg['batch_size']}")

    # 7. Set up LLM and chain
    try:
        llm = construct_llm(model_name=cfg["model_name"])
        llm_retry = llm.with_retry(
            retry_if_exception_type=(Exception,),  # Retry on any exception
            wait_exponential_jitter=True,  # Use exponential backoff with jitter
            stop_after_attempt=4,  # Max 4 retries
        )
        structured_llm = llm_retry.with_structured_output(CarveOutAssessment)
        chain = chat_prompt | structured_llm
    except Exception as e:
        logger.error(f"Failed to set up LLM chain: {e}")
        return None

    # 8. Process batches
    try:
        # Check if we should use tqdm for progress display
        try:
            from tqdm.auto import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False
            logger.info("tqdm not found, progress display will be limited")

        results = process_batches(
            chunks=chunks,
            chain=chain,
            has_tqdm=has_tqdm,
            partial_every=cfg["partial_every"],
            partial_dir=cfg["partial_dir"],
            final_path=cfg["final_path"],
            max_retries=4,
            initial_delay=5.0,
        )

        if results is not None:
            logger.info(f"Classification complete. Processed {len(results)} records.")
            return results
        else:
            logger.error("Classification failed to produce valid results.")
            return None
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        return None


def process_batches(chunks, chain, has_tqdm=False, **kwargs):
    """
    Process batches of inputs through the LLM chain.
    This is a simplified synchronous version.
    """
    import time
    from datetime import datetime
    from pathlib import Path

    partial_dir = f"{kwargs.get('partial_dir', 'partial_outputs')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    Path(partial_dir).mkdir(parents=True, exist_ok=True)
    final_path = Path(kwargs.get("final_path", "classification_results.csv"))
    final_path = final_path.with_suffix(f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    failed_chunks = []

    # Set up iterator with tqdm if available
    if has_tqdm:
        from tqdm.auto import tqdm

        chunk_iterator = tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks")
    else:
        chunk_iterator = enumerate(chunks)

    # Process each chunk
    for idx, batch in chunk_iterator:
        logger.info(f"Processing chunk {idx+1}/{len(chunks)}")
        batch_results = None

        # Retry logic
        for attempt in range(1, kwargs.get("max_retries", 4) + 1):
            try:
                batch_results = chain.batch(batch)
                logger.info(f"Chunk {idx+1} processed successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt} for chunk {idx+1} failed: {e}")
                if attempt < kwargs.get("max_retries", 4):
                    delay = kwargs.get("initial_delay", 5.0) * (2 ** (attempt - 1))
                    logger.info(f"Retrying chunk {idx+1} in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Chunk {idx+1} failed after all retries, skipping")
                    failed_chunks.append(idx)
                    batch_results = None

        # Process results
        if batch_results:
            records = []
            for i, assessment in enumerate(batch_results):
                record = assessment.dict()
                if "index" in batch[i]:
                    record["original_index"] = batch[i]["index"]
                records.append(record)
            all_records.extend(records)

        # Save partial results
        if (idx + 1) % kwargs.get("partial_every", 10) == 0 and all_records:
            df_partial = pd.DataFrame(all_records)
            partial_csv_path = Path(partial_dir) / f"partial_{idx+1}.csv"
            df_partial.to_csv(partial_csv_path, index=False)
            logger.info(f"Saved partial results to {partial_csv_path}")

        # Add a small delay to avoid overwhelming the API
        time.sleep(1)

    # Save final results
    if all_records:
        df_final = pd.DataFrame(all_records)
        df_final.to_csv(final_path, index=False)
        logger.info(f"Saved final results to {final_path}")

        if failed_chunks:
            logger.warning(f"The following chunks failed: {failed_chunks}")

        return df_final
    else:
        logger.error("No records were successfully processed")
        return None


def export_results(df=None, predictions=None):
    """
    Export processed results to Excel files.

    If df or predictions are None, will attempt to load from default paths.
    """
    # 1. Load data if not provided
    if df is None:
        df_path = CONFIG["processing"]["output_csv_path"]
        df = safe_read_csv(df_path)
        if df is None:
            logger.error(f"Failed to load processed data from {df_path}")
            return False

    if predictions is None:
        # Find the most recent predictions file
        extraction_dir = Path(EXTRACTION_DIR)
        prediction_files = list(extraction_dir.glob("classification_results*.csv"))
        if not prediction_files:
            logger.error("No prediction files found")
            return False

        # Sort by modification time, newest first
        prediction_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        newest_file = prediction_files[0]
        logger.info(f"Using most recent predictions file: {newest_file}")

        predictions = safe_read_csv(newest_file)
        if predictions is None:
            logger.error(f"Failed to load predictions from {newest_file}")
            return False

    # 2. Get company mapping for export
    _, company_mapping, _ = setup_factiva_client()

    # 3. Create company mappings with lowercase keys for the export function
    company_mapping_lower = {k.lower(): v for k, v in company_mapping.items()}

    # 4. Export results
    output_dir = CONFIG["export"]["output_dir"]
    try:
        full_df, filtered_df, targets_df = process_and_export_results(
            df, predictions, company_mapping_lower, output_dir
        )
        logger.info(f"Results exported to {output_dir}")
        logger.info(f"Full results: {len(full_df)} records")
        logger.info(f"Filtered results: {len(filtered_df)} records")
        logger.info(f"Unique targets: {len(targets_df)} records")
        return True
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return False


def main():
    """Main pipeline function that orchestrates the workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Corporate Carve-out Detection Pipeline")
    parser.add_argument("--extract", action="store_true", help="Run extraction from Factiva")
    parser.add_argument("--process", action="store_true", help="Run data processing")
    parser.add_argument("--classify", action="store_true", help="Run LLM classification")
    parser.add_argument("--export", action="store_true", help="Export formatted results")
    args = parser.parse_args()

    # If no specific steps are requested, run the full pipeline
    run_all = not (args.extract or args.process or args.classify or args.export)

    # Execute each step as requested
    df = None
    predictions = None

    if args.extract or run_all:
        logger.info("=== STEP 1: DATA EXTRACTION ===")
        job_id = extract_data()
        if not job_id:
            logger.error("Data extraction failed")
            if not run_all:
                return 1

    if args.process or run_all:
        logger.info("=== STEP 2: DATA PROCESSING ===")
        df = process_data()
        if df is None:
            logger.error("Data processing failed")
            if not run_all:
                return 1

    if args.classify or run_all:
        logger.info("=== STEP 3: LLM CLASSIFICATION ===")
        predictions = classify_data(df)
        if predictions is None:
            logger.error("LLM classification failed")
            if not run_all:
                return 1

    if args.export or run_all:
        logger.info("=== STEP 4: RESULTS EXPORT ===")
        success = export_results(df, predictions)
        if not success:
            logger.error("Results export failed")
            if not run_all:
                return 1

    logger.info("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
