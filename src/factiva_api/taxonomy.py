import pandas as pd
import logging
from typing import Dict
from factiva.news.taxonomy import Taxonomy

logger = logging.getLogger(__name__)


def init_client(user_key: str) -> Taxonomy:
    return Taxonomy(user_key=user_key)


def get_taxonomy_client(user_key: str) -> Taxonomy:
    """
    Initializes and returns a Factiva Taxonomy client.

    Args:
        user_key: Factiva API key

    Returns:
        Taxonomy client object

    Raises:
        Exception: If initialization fails
    """
    try:
        return Taxonomy(user_key=user_key)
    except Exception as e:
        logger.error(f"Failed to initialize Taxonomy client: {e}")
        raise


def get_category_codes(taxonomy_client: Taxonomy, category_name: str) -> pd.DataFrame:
    """
    Retrieves codes for a specific category (e.g., 'industries', 'companies').
    Handles potential duplicates in the index.

    Args:
        taxonomy_client: Initialized Taxonomy client
        category_name: Category to retrieve codes for (e.g., 'companies', 'industries')

    Returns:
        DataFrame with codes and descriptions for the specified category
    """
    try:
        df = taxonomy_client.get_category_codes(category_name)
        # Handle potential duplicate indices if they cause issues downstream
        if not df.index.is_unique:
            logger.warning(
                f"Duplicate indices found in '{category_name}' taxonomy. Keeping first. Note: this behavior is expected"
            )
            df = df[~df.index.duplicated(keep="first")]
        return df
    except Exception as e:
        logger.error(f"Failed to get category codes for '{category_name}': {e}")
        raise


def get_company_code_mapping(taxonomy_client: Taxonomy) -> Dict[str, str]:
    """
    Fetches company codes and returns a code -> description mapping dict.

    Args:
        taxonomy_client: Initialized Taxonomy client

    Returns:
        Dictionary mapping lowercase company codes to company names
    """
    df = get_category_codes(taxonomy_client, "companies")
    if not df.empty:
        # Ensure codes are lowercase for consistent matching if needed later
        mapping = df["description"].to_dict()
        return {str(k).lower(): v for k, v in mapping.items()}
    return {}

def get_industry_code_mapping(taxonomy_client: Taxonomy) -> Dict[str, str]:
    """
    Fetches industry codes and returns a code -> description mapping dict.

    Args:
        taxonomy_client: Initialized Taxonomy client

    Returns:
        Dictionary mapping lowercase industry codes to industry descriptions
    """
    df = get_category_codes(taxonomy_client, "industries")
    if not df.empty:
        # Ensure codes are lowercase as seen in processing_articles.ipynb usage
        mapping = df["description"].to_dict()
        return {str(k).lower(): v for k, v in mapping.items()}
    return {}


# Example usage (if run directly, requires user_key)
# if __name__ == '__main__':
#     import os
#     from dotenv import load_dotenv
#     load_dotenv('../../.env') # Adjust path as needed
#     factiva_key = os.getenv("FACTIVA_USER_KEY")
#     if not factiva_key:
#         raise ValueError("FACTIVA_USER_KEY not found in environment variables or .env file")

#     tax_client = get_taxonomy_client(factiva_key)
#     companies = get_category_codes(tax_client, 'companies')
#     print("Companies:")
#     print(companies.head())

#     company_map = get_company_code_mapping(tax_client)
#     print("\nCompany Mapping Sample:")
#     print(list(company_map.items())[:5])

#     industry_map = get_industry_code_mapping(tax_client)
#     print("\nIndustry Mapping Sample:")
#     print(list(industry_map.items())[:5])
