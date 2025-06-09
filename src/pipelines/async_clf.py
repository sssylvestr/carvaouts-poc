import argparse
import logging
from pathlib import Path
import pandas as pd
from llm_utils.factory import construct_llm, LLMChainFactory

from src.df_llm_processor import run_pipeline_sync
from src.mapping.company_mapper import map_companies
from src.schemas.models import CarveOutAssessment
from src.schemas.prompts import CARVE_OUT_ASSESSMENT_TEMPLATE, business_request

logging.basicConfig(level=logging.INFO)


def process_results(df: pd.DataFrame) -> pd.DataFrame:
     cols_mapping = {'is_co': 'is_about_carve_out',"co_confidence": "carve_out_confidence", "co_stage": "carve_out_stage",
                     "short_reasoning": "reasoning", }
     results_df = df.rename(columns=cols_mapping)
     results_df=results_df.set_index('index').sort_index()
     results_df = map_companies(results_df, col2process="target_company_code", new_col="target_company")
     results_df['target_company_code'] = results_df['target_company_code'].fillna('')
     results_df['subsidiary_company_code'] = results_df['subsidiary_company_code'].fillna('')

     return results_df

def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(description="Async articles tagging pipeline")
     parser.add_argument("--input", "-i", required=True, help="Input CSV file with articles")
     parser.add_argument("--output_dir", "-o", required=True, help="Output file for tagged articles")
     parser.add_argument("--partial_dir", "-p", default="tmp/outputs_partial", help="Directory for partial results")
     parser.add_argument("--model", "-m", default="gpt-4.1-mini", help="LLM model name")
     parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM responses")
     parser.add_argument("--test", action="store_true", help="Run in test mode with limited data")
     return parser.parse_args()

if __name__ == "__main__":
     args = parse_args()
     logging.info("Starting async articles tagging pipeline...")
     
     input_path = Path(args.input)
     df = pd.read_csv(input_path)
     if args.test:
        logging.info("Running in test mode, limiting to 100 rows")
        df = df.head(100)

     llm = construct_llm(model_name=args.model, temperature=args.temperature)
     factory = LLMChainFactory(model_name=args.model, temperature=args.temperature)
     so_chain = factory._runnable_with_structured(template=CARVE_OUT_ASSESSMENT_TEMPLATE, response_schemas=CarveOutAssessment, llm=llm, business_request=business_request)
     
     df_results = run_pipeline_sync(
          df[:],
          so_chain,
         partial_dir=args.partial_dir,
         final_path=str(Path(args.output_dir) / "raw_carveout_assessment.csv"),
         max_retries=4,
         initial_delay=3.0,
     )
     df_results = process_results(df_results)
     
     full_df = df.merge(df_results, how="left", left_index=True, right_index=True)
     full_df = full_df.rename(columns={"segment": "article_fragment"})
     full_df.to_csv(Path(args.output_dir) / "full_carveout_assessment_results.csv", index=False)

     logging.info(f"Full results saved to {args.output_dir}/full_carveout_assessment_results.csv")