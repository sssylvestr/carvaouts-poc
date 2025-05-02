import pandas as pd
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


def map_company_codes_to_names(codes_series: pd.Series, company_mapping: Dict[str, str]) -> pd.Series:
    """
    Maps a Series of comma-separated company codes to comma-separated company names.

    Args:
        codes_series: pandas Series containing comma-separated company codes (strings).
        company_mapping: Dictionary mapping uppercase company codes (str) to names (str).

    Returns:
        pandas Series containing comma-separated company names (strings).
    """

    def extract_names(company_codes_str: Optional[str]) -> str:
        if not isinstance(company_codes_str, str) or not company_codes_str:
            return ""  # Return empty string for non-string or empty input

        # Split, ensure uppercase, handle potential empty strings from split
        codes_list = [code.strip().upper() for code in company_codes_str.split(",") if code.strip()]

        company_names = []
        for code in codes_list:
            # Use .get() for safe lookup, defaulting to the code itself if not found
            company_names.append(company_mapping.get(code, code))

        return ",".join(company_names)

    logger.info(f"Mapping company codes to names for {len(codes_series)} entries.")
    return codes_series.apply(extract_names)


# Example usage (requires company_mapping dictionary):
# Assuming df['company_codes'] exists and company_map is populated
# df['company_names'] = map_company_codes_to_names(df['company_codes'], company_map)
