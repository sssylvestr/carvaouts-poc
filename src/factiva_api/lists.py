import logging
import requests

from typing import Dict, Any, List, Optional
from .common import headers, check_response, lists_url

logger = logging.getLogger(__name__)


def create_list(list_name: str, list_data: List[str], description: str = "") -> str:
    """
    Create a new Factiva list.

    Args:
        list_name: Name for the new list
        list_data: List of items to include
        description: Optional description for the list

    Returns:
        The created listId
    """
    url = f"{lists_url}/create"
    payload = {"data": {"attributes": {"listName": list_name, "listData": list_data, "description": description}}}
    logger.info(f"Creating Factiva list '{list_name}'")
    resp = requests.post(url, headers=headers, json=payload)
    data = check_response(resp)
    if data is None:
        raise RuntimeError("Failed to create list")

    list_id = data["data"]["attributes"][0]["listId"]
    logger.info(f"List '{list_name}' created with ID: {list_id}")
    return list_id


def get_list(list_id: str) -> Dict[str, Any]:
    """
    Retrieve a Factiva list by its listId.

    Args:
        list_id: ID of the list to retrieve

    Returns:
        The full list attributes dictionary
    """
    url = f"{lists_url}/{list_id}"
    logger.info(f"Retrieving Factiva list {list_id}")
    resp = requests.get(url, headers=headers)
    data = check_response(resp)
    if data is None:
        raise RuntimeError(f"Failed to retrieve list {list_id}")

    return data["data"]["attributes"]


def update_list(
    list_id: str, list_data: List[str], list_name: Optional[str] = None, description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update an existing Factiva list's contents.

    Args:
        list_id: ID of the list to update
        list_data: New list of items
        list_name: Optional new name for the list
        description: Optional new description

    Returns:
        The updated list attributes
    """
    url = f"{lists_url}/update"
    attrs: Dict[str, Any] = {"listId": list_id, "listData": list_data}
    if list_name is not None:
        attrs["listName"] = list_name
    if description is not None:
        attrs["description"] = description
    payload = {"data": {"attributes": attrs}}
    logger.info(f"Updating Factiva list {list_id}")
    resp = requests.put(url, headers=headers, json=payload)
    data = check_response(resp)
    if data is None:
        raise RuntimeError(f"Failed to update list {list_id}")

    logger.info(f"List {list_id} updated successfully.")
    return data["data"]["attributes"]


def delete_list(list_id: str) -> Dict[str, Any]:
    """
    Delete a Factiva list by its listId.

    Args:
        list_id: ID of the list to delete

    Returns:
        The API response attributes
    """
    url = f"{lists_url}/{list_id}"
    logger.info(f"Deleting Factiva list {list_id}")
    resp = requests.delete(url, headers=headers)
    data = check_response(resp)
    if data is None:
        raise RuntimeError(f"Failed to delete list {list_id}")

    logger.info(f"List {list_id} deleted successfully.")
    return data["data"]["attributes"]


def search_list_for_code(list_id: str, code: str) -> bool:
    """
    Check whether a given FCode exists in a list.

    Args:
        list_id: ID of the list to search
        code: Code to search for

    Returns:
        True if present, False otherwise
    """
    attrs = get_list(list_id)
    return code in attrs.get("listData", [])
