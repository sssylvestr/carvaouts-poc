import os
import logging
import requests
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=str(Path(__file__).parents[2] / ".env"))

# Attempt to get from environment variables first, then fallback to hardcoded
# It's strongly recommended to use environment variables for secrets
user_key = os.getenv("FACTIVA_USER_KEY", "LKbI9fY6ZEPU8RcJUxQHjwKbld52WGt0")
base_url = "https://api.dowjones.com"

headers = {"user-key": user_key, "Content-Type": "application/json", "X-API-VERSION": "3.0"}

lists_url = f"{base_url}/sns-lists"
extractions_url = f"{base_url}/extractions/documents"
taxonomy_url = f"{base_url}/taxonomy"  # Base URL for taxonomy if needed directly


def check_response(response: requests.Response) -> Dict[str, Any]:
    """
    Check if the response from an API call was successful and handle errors

    Args:
        response: Response object from requests

    Returns:
        Dict containing the JSON response if successful

    Raises:
        RuntimeError: If response indicates an error
    """
    if response.status_code in [200, 201]:
        return response.json()
    else:
        error_msg = f"API Error: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
