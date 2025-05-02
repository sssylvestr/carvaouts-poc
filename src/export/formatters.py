import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)


def process_and_export_results(
    original_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    company_mapping: Dict[str, str],
    output_dir: str = "../../demo/",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merges original data with LLM predictions, maps company codes,
    and exports full results, filtered results, and unique targets to Excel.

    Args:
        original_df: DataFrame with original article data (including index if used for joining).
        predictions_df: DataFrame with LLM predictions (including index if used for joining).
                     Expected columns: 'is_co', 'co_confidence', 'target_company_code', etc.
        company_mapping: Dictionary mapping company codes (lowercase) to names.
        output_dir: Directory to save the Excel files.

    Returns:
        Tuple containing: (full_merged_df, filtered_carveout_df, unique_targets_df)
    """
    logger.info(f"Processing and exporting results to directory: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Rename prediction columns for clarity
    cols_mapping: Dict[str, str] = {
        "is_co": "is_about_carve_out",
        "co_confidence": "carve_out_confidence",
        # Add other renames if necessary
    }
    predictions_df_renamed = predictions_df.rename(columns=cols_mapping)

    # Map target company code to name
    predictions_df_renamed["target_company"] = (
        predictions_df_renamed["target_company_code"]
        .str.lower()
        .map(company_mapping)
        .fillna(predictions_df_renamed["target_company_code"])
    )  # Fallback to code if no name found
    logger.info("Mapped target company codes to names.")

    # Merge original data with processed predictions
    # Use index for merging if predictions_df was created with original_index
    if predictions_df_renamed.index.name == "original_index":
        full_df = original_df.merge(predictions_df_renamed, how="left", left_index=True, right_index=True)
        logger.info("Merged original data and predictions using index.")
    else:
        # Fallback or alternative join strategy if indices don't match
        logger.warning(
            "Prediction DataFrame index is not 'original_index'. Attempting merge with reset index. Ensure alignment."
        )
        full_df = original_df.reset_index(drop=True).merge(
            predictions_df_renamed.reset_index(drop=True), how="left", left_index=True, right_index=True
        )

    # Export Full Results
    full_results_path = output_path / "full_results.xlsx"
    try:
        full_df.to_excel(full_results_path, index=False)
        logger.info(f"Full results saved to {full_results_path}")
    except Exception as e:
        logger.error(f"Failed to save full results to {full_results_path}: {e}")

    # Filter for Carve-outs
    filtered_df = pd.DataFrame()  # Initialize empty
    if "is_about_carve_out" in full_df.columns:
        try:
            # Attempt direct boolean indexing
            filtered_df = full_df[full_df["is_about_carve_out"] == True].copy()
        except TypeError:
            # Handle cases where the column might be object type (e.g., containing strings 'True'/'False')
            logger.warning("'is_about_carve_out' column might not be boolean. Attempting conversion.")
            try:
                # Convert common string representations to boolean
                bool_map = {"True": True, "False": False, "true": True, "false": False, 1: True, 0: False}
                filtered_df = full_df[full_df["is_about_carve_out"].map(bool_map).fillna(False)].copy()
            except Exception as conv_err:
                logger.error(
                    f"Could not reliably convert 'is_about_carve_out' to boolean: {conv_err}. Filtering might be incorrect."
                )
                filtered_df = pd.DataFrame()  # Empty DF on failure
    else:
        logger.warning("Column 'is_about_carve_out' not found. Cannot filter for carve-outs.")

    # Export Filtered Results
    if not filtered_df.empty:
        filtered_results_path = output_path / "filtered_results.xlsx"
        try:
            filtered_df.to_excel(filtered_results_path, index=False)
            logger.info(f"Filtered results saved to {filtered_results_path}")
        except Exception as e:
            logger.error(f"Failed to save filtered results to {filtered_results_path}: {e}")
    else:
        logger.info("No carve-out related articles found or filtering failed. Skipping filtered results export.")

    # Create and Export Unique Targets
    targets_df = pd.DataFrame()  # Initialize empty
    if not filtered_df.empty and all(
        col in filtered_df.columns
        for col in ["modification_date", "carve_out_confidence", "target_company_code", "target_company"]
    ):
        targets_df = (
            filtered_df[["modification_date", "carve_out_confidence", "target_company_code", "target_company"]]
            .rename(columns={"modification_date": "article_date"})
            .sort_values(by="article_date", ascending=False)
            .drop_duplicates(subset=["target_company_code"], keep="first")
            .reset_index(drop=True)
        )

        filtered_targets_path = output_path / "filtered_targets.xlsx"
        try:
            targets_df.to_excel(filtered_targets_path, index=False)
            logger.info(f"Unique filtered targets saved to {filtered_targets_path}")
        except Exception as e:
            logger.error(f"Failed to save unique targets to {filtered_targets_path}: {e}")
    elif not filtered_df.empty:
        missing_cols = [
            col
            for col in ["modification_date", "carve_out_confidence", "target_company_code", "target_company"]
            if col not in filtered_df.columns
        ]
        logger.warning(
            f"Cannot create unique targets list because filtered results are missing columns: {missing_cols}"
        )
    else:
        logger.info("Skipping unique targets export as no filtered results are available.")

    return full_df, filtered_df, targets_df
