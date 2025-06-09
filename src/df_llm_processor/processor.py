import asyncio
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from aiolimiter import AsyncLimiter

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

REQUESTS_PER_MIN = 1_000  # Azure allocation, but burst is 1–2 %
TOKENS_PER_MIN = 1_000_000

req_limiter = AsyncLimiter(REQUESTS_PER_MIN, 60)  # smooth 60-s bucket
tok_limiter = AsyncLimiter(TOKENS_PER_MIN, 60)

# initial guess for Azure’s 10-s burst bucket (will self-correct)
BURST_BUCKET = 20
burst_sem = asyncio.Semaphore(BURST_BUCKET)


async def safe_invoke(row: Dict[str, Any], chain) -> Tuple[Any, int]:
    """
    Safely invoke an LLM chain with rate limiting controls.

    This function applies three levels of rate limiting:
    1. Request-based limiting (requests per minute)
    2. Token-based limiting (tokens per minute)
    3. Burst control via semaphore (adaptive based on provider headers)

    Parameters:
    -----------
    row : Dict[str, Any]
        Dictionary containing the input data for the LLM chain.
        May include '_tok_prompt_est' for token estimation.
    chain : Any
        The LLM chain object with an ainvoke method.

    Returns:
    --------
    Tuple[Any, int]
        A tuple containing:
        - The response from the LLM chain
        - Remaining requests as reported by the API headers (-1 if not available)

    Notes:
    ------
    The function attempts to be token-efficient by releasing unused tokens
    back to the token limiter when the actual usage is less than estimated.
    """
    prompt_tok_est = row.get("_tok_prompt_est", 1_600)  # rough fallback

    async with req_limiter, tok_limiter, burst_sem:  # 3 gates
        resp = await chain.ainvoke(row)

    try:  # Reconcile token bucket (give back surplus)
        used = resp.usage.total_tokens  # OpenAI SDK style
    except Exception:
        used = prompt_tok_est
    surplus = max(prompt_tok_est - used, 0)
    if surplus:
        tok_limiter.release(surplus) # type: ignore
    # Read Azure header if present
    headers = getattr(resp, "response_metadata", {}).get("headers", {})
    remaining = int(headers.get("x-ratelimit-remaining-requests", -1))

    return resp, remaining


async def _run_row_with_retry(
    row_dict: Dict[str, Any],
    chain,
    *,
    max_retries: int,
    initial_delay: float,
):
    for attempt in range(1, max_retries + 1):
        try:
            return await safe_invoke(row_dict, chain)
        except Exception as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Row {row_dict.get('index')} failed after {max_retries} attempts: {e}") from e
            await asyncio.sleep(initial_delay * 2 ** (attempt - 1))


async def process_df_rows(
    df: pd.DataFrame,
    chain,
    *,
    partial_every: int = 5_000,
    partial_dir: str = "partials",
    final_path: str = "classification_results.csv",
    max_retries: int = 5,
    initial_delay: float = 3.0,
    cols_mapping: Dict[str, str] = {
        "source_name": "news_source",
        "title": "article_title",
        "segment": "article_body",
        "company_names": "companies",
        "company_codes": "company_codes",
        "modification_date": "modification_date",
    },
    cols2append_mapping: Dict[str, str] = {
        "company_codes": "original_company_codes",
        "companies": "original_companies",
    },
) -> pd.DataFrame:
    """
    Process DataFrame rows in parallel through an LLM chain with adaptive rate limiting.

    This function implements an efficient parallel processing system that maximizes throughput
    while respecting API rate limits. It dynamically adjusts concurrency based on provider
    capacity signals and implements automatic checkpointing to protect against data loss.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rows to be processed. Each row will be passed to the LLM chain.

    chain : Any
        An LLM chain object with an async invoke method (`ainvoke`). This chain
        should accept a dictionary of inputs and return a structured output.
        Compatible with LangChain runnables and similar interfaces.

    partial_every : int, optional
        Number of rows to process before saving a partial results checkpoint.
        Default is 5,000 rows.

    partial_dir : str, optional
        Base directory where partial results checkpoints will be saved.
        A timestamp will be appended to create a unique directory.
        Default is "partials".

    final_path : str, optional
        Base file path where the final results CSV will be saved.
        A timestamp will be appended to the filename.
        Default is "classification_results.csv".

    max_retries : int, optional
        Maximum number of retry attempts for failed row processing.
        Uses exponential backoff strategy between retries.
        Default is 5 attempts.

    initial_delay : float, optional
        Initial delay in seconds before the first retry attempt.
        Subsequent retries will use exponential backoff (delay * 2^attempt).
        Default is 3.0 seconds.

    cols_mapping : Dict[str, str], optional
        Mapping from DataFrame column names to the expected input keys for the LLM chain.
        Keys are the DataFrame column names, values are the desired keys in chain input.
        Default maps standard news article fields to common prompt input names.

    cols2append_mapping : Dict[str, str], optional
        Mapping of original columns to preserve in the results DataFrame.
        Keys are the original column names, values are the new column names in the results.
        Default preserves company codes and names with "original_" prefix.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the structured outputs from the LLM chain for all
        successfully processed rows. Includes the original index and any columns
        specified in cols2append_mapping.

    Notes
    -----
    Implementation details:

    - Uses three-level rate limiting: requests per minute, tokens per minute, and burst control
    - Adaptive burst capacity that self-adjusts based on provider rate limit headers
    - Task scheduling with dynamic concurrency
    - Failed rows are tracked but don't stop overall processing
        
    When using with structured output chains, the chain should return an object with a
    model_dump() method (like Pydantic models) or a dictionary.

    Examples
    --------
    ```python
    from pydantic import BaseModel, Field
    from llm_utils.factory import LLMChainFactory

    # Define your output schema
    class CarveOutAssessment(BaseModel):
        is_relevant: bool = Field(description="Is the article relevant?")
        score: float = Field(description="Relevance score from 0-1")

    # Create a chain
    factory = LLMChainFactory(model_name="gpt-4-mini")
    chain = factory.build_structured_output_chain(
        template="Analyze this news: {article_body}",
        output_schema=CarveOutAssessment
    )

    # Process the DataFrame
    results_df = await process_df_rows(
        news_df,
        chain,
        partial_dir="outputs/project_news/partials",
        final_path="outputs/project_news/results"
    )
    ```

    With custom column mapping:

    ```python
    results_df = await process_df_rows(
        customer_df,
        sentiment_chain,
        cols_mapping={
            "feedback_text": "text",
            "product_name": "product",
            "purchase_date": "date"
        },
        cols2append_mapping={
            "customer_id": "original_id",
            "feedback_source": "source"
        }
    )
    ```
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(f"{partial_dir}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataframe → list[dict]
    rows: List[Dict[str, Any]] = df.rename(columns=cols_mapping).reset_index().to_dict(orient="records") #type: ignore

    logger.debug(f"Processing {len(rows)} rows")

    results, failures = [], []
    pbar = tqdm(total=len(rows), desc="Rows processed", unit="row")

    pending: List[asyncio.Task] = []  # internal queue of scheduled tasks

    async def spawn(row_dict):
        t = asyncio.create_task(
            _run_row_with_retry(
                row_dict,
                chain,
                max_retries=max_retries,
                initial_delay=initial_delay,
            )
        )
        t.row_meta = row_dict # type: ignore
        pending.append(t)

    # prime pump with burst_bucket tasks
    for row in rows[:BURST_BUCKET]:
        await spawn(row)
    row_iter = iter(rows[BURST_BUCKET:])
    logging.debug(f"Primed with {len(pending)} tasks")

    while pending:
        # wait for the next task to finish
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            logging.debug(f"Task {task} done")
            pending.remove(task)
            remaining = -1  # Default value if task.result() fails
            try:
                resp, remaining = task.result()
                logging.debug(f"Task {task} result: {resp}")
                rec = resp.model_dump() if hasattr(resp, "model_dump") else resp
                for col, new_col in cols2append_mapping.items():
                    rec[new_col] = task.row_meta.get(col, "") #type: ignore
                rec["index"] = task.row_meta.get("index") #type: ignore # index must always be present
                results.append(rec)
            except Exception:
                logging.error(f"Task {task} failed: {task.exception()}")
                failures.append(task.row_meta.get("index")) #type: ignore

            pbar.update(1)

            # ---- adaptive part: widen/narrow the burst semaphore ----
            if remaining >= 0:
                # we finished one, so one slot is free; add any EXTRA free slots
                extra = max(remaining - burst_sem._value, 0) #type: ignore
                for _ in range(extra):
                    burst_sem.release()

            if len(results) and len(results) % partial_every == 0:  # periodic checkpoint
                pd.DataFrame(results).to_csv(out_dir / f"partial_{len(results)}.csv", index=False)
                print(f"[checkpoint] {len(results)} rows → {out_dir}")

            try:  # schedule next row if we have more data
                next_row = next(row_iter)
                await spawn(next_row)
            except StopIteration:
                pass  # no more rows to enqueue

    pbar.close()

    logger.info(f"Processing complete: {len(results)} rows processed")

    # final save
    df_final = pd.DataFrame(results)
    final_csv = Path(f"{final_path.strip('.csv')}_{stamp}.csv")
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(final_csv, index=False)
    logger.info(f"[done] {len(results)} rows saved → {final_csv}")

    if failures:
        logger.warning(f"⚠️  {len(failures)} rows failed: {failures[:10]} …")

    return df_final

def run_pipeline_sync(*args, **kwargs):
    return asyncio.run(process_df_rows(*args, **kwargs))

async def run_pipeline(*args, **kwargs):
    """
    Convenience function to run the DataFrame processing pipeline.

    This is a thin wrapper around process_df_rows that maintains the same
    interface but with a more intuitive name for end users.

    Parameters:
    -----------
    *args
        Positional arguments passed to process_df_rows.
    **kwargs
        Keyword arguments passed to process_df_rows.

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame with LLM responses.

    Examples:
    ---------
    >>> df_results = await run_pipeline(
    ...     df,
    ...     so_chain,
    ...     partial_dir="outputs/partial",
    ...     final_path="outputs/final_results",
    ...     max_retries=4,
    ...     initial_delay=3.0,
    ... )
    """
    return await process_df_rows(*args, **kwargs)