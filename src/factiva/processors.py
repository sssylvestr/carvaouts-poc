import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


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

    df_processed = df.drop(columns=cols_to_drop, errors="ignore")
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
                # Assuming timestamps are in milliseconds since epoch
                df_processed[col] = pd.to_datetime(df_processed[col], unit="ms", errors="coerce")
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
            )
            df_processed[col] = df_processed[col].fillna("")

        # Combine all company code columns into one, ensuring uniqueness and sorting
        df_processed["all_company_codes"] = df_processed[cc_columns].agg(
            lambda x: ",".join(sorted(list(set(",".join(x).split(",")) - {""}))), axis=1
        )
        df_processed = df_processed.drop(columns=cc_columns, errors="ignore")
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

    # Ensure all desired columns exist, add missing ones with default value (e.g., None or NaN)
    for col in final_columns_order:
        if col not in df_processed.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, adding as empty.")
            df_processed[col] = None  # Or pd.NA or appropriate default

    df_processed = df_processed[final_columns_order]
    logger.info(f"Reordered columns. Final shape: {df_processed.shape}")

    return df_processed


def limit_text_length(df: pd.DataFrame, column: str = "body", max_length: int = 2000) -> pd.DataFrame:
    """
    Truncates text in a specified column to a maximum length.

    Args:
        df: DataFrame containing text data
        column: Column name to truncate
        max_length: Maximum character length

    Returns:
        DataFrame with truncated text column
    """
    if column in df.columns:
        original_lengths = df[column].str.len()
        df[column] = df[column].str[:max_length]
        new_lengths = df[column].str.len()
        truncated_count = (original_lengths > max_length).sum()
        if truncated_count > 0:
            logger.info(f"Truncated {truncated_count} entries in column '{column}' to {max_length} characters.")
    else:
        logger.warning(f"Column '{column}' not found for truncation.")
    return df


def drop_body_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate rows based on the 'body' column and resets index.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with duplicates removed
    """
    if "body" in df.columns:
        initial_rows = len(df)
        df.drop_duplicates(subset=["body"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        dropped_count = initial_rows - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} duplicate rows based on 'body' column.")
    else:
        logger.warning("Column 'body' not found for duplicate removal.")
    return df
