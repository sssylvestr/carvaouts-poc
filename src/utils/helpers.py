"""
Utility helper functions for the carveouts detection project.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, TypeVar, Union, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


def chunk_list(items: List[T], batch_size: int) -> List[List[T]]:
    """
    Split a list into sublists (batches) of maximum length `batch_size`.

    Args:
        items: The list to split
        batch_size: Maximum size of each batch

    Returns:
        List of batches (lists)
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure that the directory at the specified path exists, creating it if needed.

    Args:
        path: Directory path as string or Path object

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_read_csv(path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read a CSV file, handling exceptions.

    Args:
        path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame if successful, None on failure
    """
    try:
        logger.info(f"Reading CSV from {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Successfully read {len(df)} rows from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV {path}: {e}")
        return None


def safe_write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> bool:
    """
    Safely write a DataFrame to a CSV file, ensuring directory exists.

    Args:
        df: DataFrame to save
        path: Path where the CSV will be saved
        **kwargs: Additional arguments for df.to_csv

    Returns:
        True if successful, False otherwise
    """
    try:
        path_obj = Path(path)
        ensure_directory_exists(path_obj.parent)
        df.to_csv(path_obj, **kwargs)
        logger.info(f"Successfully wrote {len(df)} rows to {path}")
        return True
    except Exception as e:
        logger.error(f"Error writing CSV to {path}: {e}")
        return False


def file_age_in_days(path: Union[str, Path]) -> Optional[float]:
    """
    Get the age of a file in days.

    Args:
        path: Path to the file

    Returns:
        Age in days if file exists, None otherwise
    """
    import time
    from datetime import datetime

    try:
        mtime = os.path.getmtime(path)
        now = time.time()
        return (now - mtime) / (60 * 60 * 24)  # Convert seconds to days
    except FileNotFoundError:
        logger.warning(f"File not found when checking age: {path}")
        return None
    except Exception as e:
        logger.error(f"Error checking file age for {path}: {e}")
        return None


def get_latest_file(directory: Union[str, Path], pattern: str) -> Optional[Path]:
    """
    Get the most recently modified file matching a pattern in the directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match files

    Returns:
        Path to the latest file, or None if no files match
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return None

    matching_files = list(dir_path.glob(pattern))
    if not matching_files:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
        return None

    # Sort by modification time, newest first
    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Latest file matching '{pattern}': {latest_file}")
    return latest_file


def retry_with_backoff(
    func: Callable, max_retries: int = 3, initial_delay: float = 1.0, jitter: bool = True
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        func: Function to wrap
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        jitter: Whether to add random jitter to the delay

    Returns:
        Wrapped function with retry logic
    """
    import time
    import random
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                    raise

                # Calculate delay with optional jitter
                actual_delay = delay
                if jitter:
                    actual_delay = delay * (0.8 + 0.4 * random.random())  # ±20% jitter

                logger.warning(
                    f"Attempt {attempt+1}/{max_retries+1} for {func.__name__} failed: {e}. "
                    f"Retrying in {actual_delay:.2f}s..."
                )
                time.sleep(actual_delay)
                delay *= 2  # Exponential backoff

    return wrapper


def get_timestamp_str() -> str:
    """
    Get a timestamp string for file naming.

    Returns:
        Formatted timestamp string
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_timestamped_filename(base_path: Union[str, Path], suffix: str = "") -> Path:
    """
    Create a filename with a timestamp.

    Args:
        base_path: Base file path
        suffix: Optional suffix to add before file extension

    Returns:
        Path with timestamp inserted before the extension
    """
    path_obj = Path(base_path)
    stem = path_obj.stem
    extension = path_obj.suffix

    timestamp = get_timestamp_str()
    new_filename = f"{stem}_{timestamp}{suffix}{extension}"

    return path_obj.parent / new_filename
