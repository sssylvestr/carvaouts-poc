import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, TypeVar, Union, Callable
from tqdm import tqdm

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


def read_dataset(dir_: str, batch_rows: int = 250_000, string_cols=None) -> pd.DataFrame:
    """
    Stream an entire partitioned Parquet dataset into a pandas DataFrame,
    keeping Arrow dtypes to stay memory-efficient except for specified string columns.

    Parameters:
    -----------
    dir_ : str
        Path to parquet file or directory
    batch_rows : int
        Number of rows to read in each batch
    string_cols : list or None
        Columns to load as Python strings instead of Arrow strings
    """
    import pyarrow.dataset as ds
    import pyarrow as pa

    # Default to segments if no string_cols provided
    if string_cols is None:
        string_cols = ["segments"]
    elif isinstance(string_cols, str):
        string_cols = [string_cols]

    # First, read the schema to get column names
    dataset = ds.dataset(dir_, format="parquet")
    schema = dataset.schema
    column_names = schema.names

    # Identify which columns should be converted to Python strings
    convert_to_str = [name for name in column_names if name in string_cols]

    batches = []

    # Create a custom type mapper function that checks column name
    def custom_types_mapper(pa_type, pa_field_name):
        if pa.types.is_string(pa_type) and pa_field_name in convert_to_str:
            return str
        return pd.ArrowDtype(pa_type)

    scanner = dataset.scanner(batch_size=batch_rows, use_threads=True)

    for rb in scanner.to_batches():
        # Process each batch and apply the custom type mapping
        df_batch = rb.to_pandas(
            types_mapper=lambda pa_type, pa_field_name=None: custom_types_mapper(pa_type, pa_field_name)
        )
        batches.append(df_batch)

    if not batches:
        # Return empty DataFrame with schema if no data
        return pd.DataFrame()

    return pd.concat(batches, ignore_index=True, copy=False)


def write_dataset(out_dir: str, df: pd.DataFrame) -> None: #TODO: optimize data types; reduce batch_size according to your RAM
    import gc, os
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(8)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    batch = 100_000  # rows per Arrow RecordBatch

    for start in tqdm(range(0, len(df), batch)):
        tbl = pa.Table.from_pandas(df.iloc[start : start + batch], preserve_index=False)
        pq.write_to_dataset(
            tbl,
            root_path=out_dir,
            partition_cols=["year", "month"],
            compression="zstd",
            existing_data_behavior="overwrite_or_ignore"
        )
        del tbl
        gc.collect()
