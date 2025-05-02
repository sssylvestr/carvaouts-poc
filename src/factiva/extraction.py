import requests
import json
import logging
import os
import time
from time import sleep
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from fastavro import reader
import pandas as pd
from pprint import pprint

# Import from local modules
from .common import headers, base_url, check_response, lists_url, extractions_url

logger = logging.getLogger(__name__)

# === List Management Functions ===


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


# === Extraction Functions ===


def create_extraction(query_body: Dict[str, Any]) -> str:
    """
    Submit a new Snapshot Extraction job.

    Args:
        query_body: Query parameters as a dictionary

    Returns:
        Job ID for the created extraction
    """
    url = f"{extractions_url}"
    logger.info("Creating extraction job...")
    resp = requests.post(url, headers=headers, json=query_body)
    data = check_response(resp)
    if data is None:
        raise RuntimeError("Failed to create extraction job")

    job_id = data["data"]["id"]
    logger.info(f"Extraction job created: {job_id}")
    return job_id


def check_extraction_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the current state of an extraction job.

    Args:
        job_id: ID of the extraction job

    Returns:
        Full JSON response on success, None otherwise
    """
    url = f"{extractions_url}/{job_id}"
    resp = requests.get(url, headers=headers)
    return check_response(resp)


def wait_for_extraction_completion(job_id: str, max_retries: int = 60, initial_sleep: int = 10) -> Dict[str, Any]:
    """
    Poll the extraction job until it reaches JOB_STATE_DONE or JOB_STATE_FAILED.

    Args:
        job_id: ID of the extraction job
        max_retries: Maximum number of status checks
        initial_sleep: Initial delay between checks in seconds

    Returns:
        Job attributes on success

    Raises:
        RuntimeError: If the job fails
        TimeoutError: If max_retries is reached
    """
    sleep_time = initial_sleep
    for attempt in range(1, max_retries + 1):
        status_response = check_extraction_status(job_id)
        if not status_response:
            raise RuntimeError(f"Failed to get status for job {job_id} after API error.")

        state = status_response["data"]["attributes"]["current_state"]
        logger.info(f"[Job {job_id}][Attempt {attempt}/{max_retries}] State: {state}")

        if state == "JOB_STATE_DONE":
            logger.info(f"Extraction job {job_id} completed successfully.")
            return status_response["data"]["attributes"]
        if state == "JOB_STATE_FAILED":
            error_detail = "Unknown error"
            if "errors" in status_response and status_response["errors"]:
                error_detail = status_response["errors"][0].get("detail", "No detail provided")
            raise RuntimeError(f"Extraction job {job_id} failed: {error_detail}")

        logger.info(f"Sleeping for {sleep_time}s before retry... (Job: {job_id})")
        sleep(sleep_time)
        sleep_time = min(sleep_time * 1.5, 300)  # Cap back-off at 5 minutes

    raise TimeoutError(f"Job {job_id} did not complete after {max_retries} attempts.")


def download_files(files: List[Dict[str, Any]], snapshot_id: str, target_root: str = "../../extractions") -> None:
    """
    Stream-download each file URI into target_root/{snapshot_id}/.

    Args:
        files: List of file metadata dictionaries with 'uri' key
        snapshot_id: ID of the extraction job
        target_root: Base directory for downloads
    """
    download_dir = Path(target_root) / snapshot_id
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {len(files)} files into '{download_dir.resolve()}'")

    for file_meta in files:
        uri = file_meta["uri"]
        # Use Path object for robust path handling
        file_name = Path(uri).name
        dest_path = download_dir / file_name
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

        logger.info(f"-> Downloading {file_name}")

        for attempt in range(1, 4):  # Retry up to 3 times
            try:
                with requests.get(uri, headers=headers, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp_path, "wb") as tmpf:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                tmpf.write(chunk)
                # Atomic rename on success
                os.replace(tmp_path, dest_path)
                logger.info(f"   Saved to {dest_path}")
                break  # Success, exit retry loop
            except Exception as e:
                logger.warning(f"   Attempt {attempt}/3 failed for {file_name}: {e}")
                if tmp_path.exists():  # Clean up temp file on failure
                    try:
                        tmp_path.unlink()
                    except OSError as unlink_err:
                        logger.error(f"    Could not remove temp file {tmp_path}: {unlink_err}")

                if attempt == 3:
                    logger.error(f"   Giving up on {file_name} after 3 attempts.")
                else:
                    sleep(5 * attempt)  # Wait before retrying


def run_extraction(query_body: Dict[str, Any], target_root: str = "../../extractions") -> str:
    """
    Full end-to-end extraction: submit job, wait, download files.

    Args:
        query_body: Query parameters as a dictionary
        target_root: Base directory for downloads

    Returns:
        Job ID of the extraction
    """
    job_id = create_extraction(query_body)
    try:
        attrs = wait_for_extraction_completion(job_id)
        files = attrs.get("files", [])
        if not files:
            logger.warning(f"No files found for completed job {job_id}")
        else:
            download_files(files, job_id, target_root=target_root)
        logger.info(f"Extraction {job_id} processing complete.")
    except (RuntimeError, TimeoutError) as e:
        logger.error(f"Extraction process failed for job {job_id}: {e}")
    return job_id


# === Avro Reading Functions ===


def read_avro_file(path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Yield each record (as a dict) from an Avro file.

    Args:
        path: Path to the Avro file

    Yields:
        Record dictionaries from the Avro file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    logger.info(f"Reading Avro file: {path}")
    try:
        with open(path, "rb") as fo:
            avro_reader = reader(fo)
            for record in avro_reader:
                yield record
    except FileNotFoundError:
        logger.error(f"Avro file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Avro file {path}: {e}")
        raise


def avro_to_dataframe(avro_path: str) -> pd.DataFrame:
    """
    Reads an Avro file and converts it into a pandas DataFrame.

    Args:
        avro_path: Path to the Avro file

    Returns:
        DataFrame containing the records
    """
    records = list(read_avro_file(avro_path))
    if not records:
        logger.warning(f"No records found in Avro file: {avro_path}")
        return pd.DataFrame()
    logger.info(f"Read {len(records)} records from {avro_path}. Converting to DataFrame.")
    return pd.DataFrame.from_records(records)


def avro_to_csv(avro_path: str, csv_path: str, **kwargs) -> None:
    """
    Reads an Avro file and saves it as a CSV.

    Args:
        avro_path: Path to the Avro file
        csv_path: Path where the CSV will be saved
        **kwargs: Additional arguments to pass to df.to_csv()
    """
    df = avro_to_dataframe(avro_path)
    if not df.empty:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, **kwargs)
        logger.info(f"Converted {avro_path} to CSV: {csv_path}")
    else:
        logger.warning(f"Skipping CSV creation for empty Avro file: {avro_path}")


# === Explain Functions ===


def create_explain(query_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Create an explain request to check how many documents match a query.

    Args:
        query_body: Query parameters as a dictionary

    Returns:
        JSON response on success, None on failure
    """
    url = f"{base_url}/extractions/documents/_explain"
    response = requests.post(url, headers=headers, data=json.dumps(query_body))
    return check_response(response)

def check_explain_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Check the status of an explain job.

    Args:
        job_id: ID of the explain job

    Returns:
        JSON response on success, None on failure
    """
    url = f"{base_url}/extractions/documents/{job_id}/_explain"
    response = requests.get(url, headers=headers)
    return check_response(response)

def wait_for_explain_completion(job_id: str, max_retries: int = 100, sleep_time: int = 10) -> Optional[Dict[str, Any]]:
    """
    Wait for an explain job to complete.

    Args:
        job_id: ID of the explain job
        max_retries: Maximum number of status checks
        sleep_time: Time to wait between checks in seconds

    Returns:
        Final job status on success, None on failure
    """
    for i in range(max_retries):
        status = check_explain_status(job_id)
        if not status:
            return None
            
        current_state = status['data']['attributes']['current_state']
        logger.info(f"Current state: {current_state}")
        
        if current_state == "JOB_STATE_DONE":
            return status
        
        if i < max_retries - 1:
            logger.info(f"Waiting {sleep_time} seconds...")
            sleep(sleep_time)
    
    logger.warning("Max retries reached. Job might still be running.")
    return None

def run_explain(input_query: Dict[str, Any], num_samples: int = 5) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Run full explain workflow: create job, wait for completion, get samples.

    Args:
        input_query: Query parameters as a dictionary
        num_samples: Number of sample documents to request

    Returns:
        Tuple of (final_status, samples) on success, (None, None) on failure
    """
    explain_result = create_explain(input_query)

    if explain_result:
        logger.info("\nExplain job created:")
        job_id = explain_result['data']['id']
        logger.info(f"Job ID: {job_id}")
        
        final_status = wait_for_explain_completion(job_id)

        logger.info(f"\nGetting samples for job ID: {job_id}")
        if final_status and 'counts' in final_status['data']['attributes']:
            count = final_status['data']['attributes']['counts']
            logger.info(f"\nNumber of documents matching the query: {count}")
            
            samples_url = f"{base_url}/extractions/samples/{job_id}?num_samples={num_samples}"
            samples_response = requests.get(samples_url, headers=headers)
            samples = samples_response.json()
            
            return final_status, samples
    
    return None, None

def run_extraction_workflow(query_body: Dict[str, Any]) -> str:
    """
    Full end-to-end extraction: submit job, wait, download files.

    Args:
        query_body: Query parameters as a dictionary

    Returns:
        Job ID of the completed extraction
    """
    job_id = create_extraction(query_body)
    attrs = wait_for_extraction_completion(job_id)
    files = attrs.get("files", [])
    if not files:
        logger.warning(f"No files found for job {job_id}")
    else:
        download_files(files, job_id)
    logger.info(f"Extraction {job_id} complete.")
    return job_id

def read_avro_file(path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Yield each record (as a dict) from an Avro file.
    
    Args:
        path: Path to the Avro file
    
    Yields:
        Each record as a dictionary
    """
    with open(path, 'rb') as fo:
        avro_reader = reader(fo)
        for record in avro_reader:
            yield record

def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of records to a pandas DataFrame.
    
    Args:
        records: List of record dictionaries
    
    Returns:
        DataFrame containing all records
    """
    return pd.DataFrame(records)
