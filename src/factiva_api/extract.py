import logging, os
import requests
from time import sleep
from typing import Dict, Any, List, Optional
from pathlib import Path

from .common import headers, check_response, extractions_url

logger = logging.getLogger(__name__)


def create_extraction(query_body: Dict[str, Any]) -> str:
    """
    Submit a new Snapshot Extraction job.

    Args:
        query_body: Query parameters as a dictionary

    Returns:
        Job ID for the created extraction
    """
    logger.info("Creating extraction job...")
    resp = requests.post(extractions_url, headers=headers, json=query_body)
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
            detail = status_response["errors"][0].get("detail", "No detail provided")
            raise RuntimeError(f"Extraction job {job_id} failed: {detail}")

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
                    with open(tmp_path, "wb") as tmp_file:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                tmp_file.write(chunk)
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
