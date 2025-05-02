import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def map_industry_codes_to_names(codes_series: pd.Series, industry_mapping: Dict[str, str]) -> pd.Series:
    """
    Maps a Series of comma-separated industry codes to comma-separated industry descriptions.

    Args:
        codes_series: pandas Series containing comma-separated industry codes (strings).
        industry_mapping: Dictionary mapping lowercase industry codes (str) to descriptions (str).

    Returns:
        pandas Series containing comma-separated industry descriptions (strings).
    """

    def extract_descriptions(industry_codes_str: Optional[str]) -> str:
        if not isinstance(industry_codes_str, str) or not industry_codes_str:
            return ""  # Return empty string for non-string or empty input

        # Split, ensure lowercase, sort, handle potential empty strings
        codes_list = sorted(list(set(code.strip().lower() for code in industry_codes_str.split(",") if code.strip())))

        industry_descriptions = []
        for code in codes_list:
            # Use .get() for safe lookup, defaulting to the code itself if not found
            industry_descriptions.append(industry_mapping.get(code, code))

        return ",".join(industry_descriptions)

    logger.info(f"Mapping industry codes to names for {len(codes_series)} entries.")
    return codes_series.apply(extract_descriptions)


# Example usage (requires industry_mapping dictionary):
# Assuming df['industry_codes'] exists and industry_map is populated
# df['industries'] = map_industry_codes_to_names(df['industry_codes'], industry_map)
