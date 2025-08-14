import argparse
import logging
from pathlib import Path
import pandas as pd
from langchain_core.prompts import PromptTemplate

from llm_utils.factory import LLMChainFactory
from src.df_llm_processor import run_pipeline_sync
from src.schemas.models import (
    CarveOutIdentificationSummary,
    SearchCarveOutIdentificationSummary,
)
from src.schemas.prompts import (
    IDENTIFICATION_PROMPT_TEMPLATE,
    IDENTIFICATION_SEARCH_PROMPT_TEMPLATE,
    business_request,
)
from src.utils.excel_export import export_summary_to_excel

def most_common(series):
    return series.value_counts().index[0] if not series.empty else None

# Function to concatenate unique values with optional limit
def set_concat(series, max_items=None, max_chars=None):
    """
    Concatenate unique values with limits on items and characters
    """
    unique_values = pd.Series(series).dropna().astype(str).unique()
    
    # Apply item limit if specified
    if max_items and len(unique_values) > max_items:
        truncated = unique_values[:max_items]
        result = " | ".join(truncated) + f" (+ {len(unique_values) - max_items} more)"
    else:
        result = " | ".join(unique_values) if len(unique_values) > 0 else ""
    
    # Apply character limit if specified
    if max_chars and len(result) > max_chars:
        result = result[:max_chars] + "..."
        
    return result

logging.basicConfig(level=logging.INFO)

def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(description="Async articles tagging pipeline")
     parser.add_argument("--input", "-i", required=True, help="Input CSV file with classification results")
     parser.add_argument("--output_dir", "-o", required=True, help="Output dir for final outputs")
     parser.add_argument("--partial_dir", "-p", default="tmp/summary_outputs_partial", help="Directory for partial results")
     parser.add_argument("--test", action="store_true", help="Run in test mode with limited data")
     return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.info("Starting extraction of carve-out targets...")

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    logging.info(f"Loaded data from {input_path} with shape: {df.shape}")
    df = df[df.is_relevant & df.is_about_carve_out]  # Filter relevant articles
    logging.info(
        f"Filtered relevant articles classified as carve-outs with shape: {df.shape}"
    )
    if args.test:
        logging.info("Running in test mode, limiting to 200 rows")
        df = df.head(200)

     factory = LLMChainFactory(model_name="o4-mini", provider="azure")
     summary_chain = factory.runnable_with_pydantic(
          template=IDENTIFICATION_PROMPT_TEMPLATE,
          pydantic_model=CarveOutIdentificationSummary,
          llm=factory.llm,
          business_request=business_request,
     )

     logging.info("Loaded data and created chain for carve-out identification")

     summary_cols_mapping = {
          "company_names": "companies",
          "company_codes": "company_codes",
          "modification_date": "date",
     }

     df_results = run_pipeline_sync(
          df,
          summary_chain,
          partial_dir=args.partial_dir,
          final_path=str(Path(args.output_dir) / "summary_results.csv"),
          cols_mapping=summary_cols_mapping,
          cols2append_mapping={
               "source_name": "source_name",
               "article_fragment": "article_fragment",
               "date": "date",
          },
          max_retries=4,
          initial_delay=3.0,
          partial_every=2800,
     )

     required_fields = {"source_name", "article_fragment", "date"}
     if not required_fields.issubset(df_results.columns):
          missing = required_fields - set(df_results.columns)
          raise ValueError(
               f"Missing required fields in df_results: {', '.join(missing)}"
          )

     search_chain = factory.build_search_runnable_with_structured_output(
          pydantic_model=SearchCarveOutIdentificationSummary,
          model="gpt-5",
     )
     search_template = PromptTemplate.from_template(
          template=IDENTIFICATION_SEARCH_PROMPT_TEMPLATE
     )
     search_chain = search_template | search_chain

     search_cols_mapping = {
          "source_name": "news_source",
          "article_fragment": "article_body",
          "target_company": "target_company",
          "potential_disposal": "potential_disposal",
          "potential_disposal_company": "potential_disposal_company",
     }

     # Process search enrichment using the same synchronous wrapper.  The
     # run_pipeline_sync helper handles all asynchronous execution and
     # checkpointing, so we don't manage tasks or progress bars here.
     df_search = run_pipeline_sync(
          df_results,
          search_chain,
          partial_dir=args.partial_dir,
          final_path=str(Path(args.output_dir) / "search_results.csv"),
          cols_mapping=search_cols_mapping,
          max_retries=4,
          initial_delay=3.0,
          partial_every=2800,
     )

     df_results = df_results.drop(
          columns=["source_name", "article_fragment", "potential_disposal_company"],
          errors="ignore",
     )
     df_results = df_results.merge(df_search, on="index", how="left")
     df_results = df_results.rename(
          columns={
               "financial_group_hq": "group_hq",
               "group_vertical": "vertical",
               "potential_disposal_industry": "disposal_nc_sector",
          }
     )

     logging.info(
          "Extraction completed, merging full results with original data"
     )

     df_results = df_results[df_results.relevant.fillna(False)]
     df_results = df_results.set_index("index").sort_index()

     full_df = df[
          [
               "source_name",
               "title",
               "article_fragment",
               "carve_out_stage",
               "reasoning",
          ]
     ].merge(
          df_results,
          left_index=True,
          right_index=True,
          how="left",
          suffixes=["_original", "_new"],
     )  # type: ignore

     grouped_summary = full_df.groupby("target_company").agg(
          {
               "group": most_common,  # Most popular group
               "group_hq": most_common,  # Most common HQ
               "vertical": most_common,  # Most common vertical
               "potential_disposal": lambda x: set_concat(x, 4),
               "potential_disposal_company": lambda x: set_concat(x, 4),
               "potential_disposal_country": lambda x: set_concat(x, 4),
               "disposal_nc_sector": lambda x: set_concat(x, 4),
               "potential_disposal_vertical": lambda x: set_concat(x, 4),
               "rationale": lambda x: set_concat(x, 4),
               "date": "min",
               "interest_score": "mean",
               "carve_out_stage": most_common,
               "source_name": lambda x: set_concat(x, 4),
               "title": lambda x: set_concat(x, 4),
               "article_quote": lambda x: set_concat(x, 4),
               "article_fragment": lambda x: set_concat(
                    x, max_items=4, max_chars=2500
               ),
               "reasoning": lambda x: set_concat(x, 4),
          }
     )

     grouped_summary["interest_score"] = grouped_summary["interest_score"].round(2)
     grouped_summary["date"] = pd.to_datetime(grouped_summary["date"]).dt.strftime(
          "%Y-%m-%d"
     )
     grouped_summary = grouped_summary.loc[
          :, grouped_summary.columns != "reasoning"
     ].reset_index()

     export_path = export_summary_to_excel(
          grouped_summary, excel_path=Path(args.output_dir) / "carveouts_summary.xlsx"
     )  # type: ignore
     logging.info(f"Exported summary to {export_path}")
     logging.info("Pipeline completed successfully!")