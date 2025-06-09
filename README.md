# Carve-outs POC

PoC is dedicated to detecting companies that may be potentially interested in doing a corporate carve-out.
Wider case is covering building a separate pipeline that would aggregate and process articles from different sources and detect various news signals that may be of interest for Investment Professionals.

## External Sources

Current:

* Factiva News

TBA:

* Factset Transcripts (API/DataBricks)
* Mergermarket (M&A)

## Factiva extraction pipeline overview

Using Factiva News API, we extract articles relevant to business request (carve-out) for now, filter results for relevancy and prepare a list of targets - companies interested in carve-outs

Pipeline description:  

1. Form extraction query using /explain endpoint
2. Run extraction
3. Processing articles:
    * Initial cleaning (.avro -> .csv/parquet -> cleaning)
    * Splitting very large articles into smaller ones using [TextTilingAlgorithm by NLTK](https://www.nltk.org/api/nltk.tokenize.texttiling.html)
    * De-duplicating smaller articles (hash-based -> semantic deduplication)
    * Classifying resulting set of articles for relevance to a business case
    * Extracting names of target companies from relevant articles together with additional info
4. Post-processing and preparing a final list of articles and target companies for deal team.
5. TBA: additional extraction of more details about target companies

## How to run?

1. Installation:
    **Using UV:**

    ```bash
    uv venv  
    source .venv/bin/activate
    uv sync --extra-index-url $PIP_EXTRA_INDEX_URL #basic installation
    uv sync --extra-index-url $PIP_EXTRA_INDEX_URL --all-extras #extras: semantic similarity; jupyter
    ```

    **Using pip:**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

2. Running the pipeline:

    **Automated pipeline**  
    The easiest way to run the full pipeline is using the provided script:

    ```bash
    # Basic run
    ./run_pipeline.sh extractions/job-id/result.csv

    # With specific model
    ./run_pipeline.sh extractions/job-id/result.csv gpt-4.1

    # Test mode with smaller datasets
    ./run_pipeline.sh extractions/job-id/result.csv --test

    # Continue from existing directory
    ./run_pipeline.sh extractions/job-id/result.csv - processing_20250606_120717
    ```

    **Step-by-Step Execution**
    Alternatively, you can run each step individually:

    1. Form extraction query and execute it; download data (be very careful due to quotas! - better to run it in a notebook ): `src/notebooks/data_extraction.ipynb`
        Input: None
        Output: extractions/{job_id}/result.csv
    2. Preprocess and tile articles into smaller ones (optimized):

        ```python
        python -m src.pipelines.preprocess_and_tile -i {extraction_csv} -o {output_dir}
        ```

        Input: extractions/{job_id}/result.csv  
        Output: {output_dir} with a set of .parquet files

    3. Deduplicate using non-semantic hashing methods (not optimized):

        ```python -m src.pipelines.hash_and_dedup -i {output_dir} -o {output_file}```

        Input: {output_dir} with a set of .parquet files  
        Output: {output_file} (parquet)

    4. Semantic deduplication (not optimized):

        ```python
        python -m src.pipelines.ssim_dedup -i {output_file} -o {ssim_output_file}
        ```

    5. Merging deduplicated article segments with useful metadata:

        ```python
        python -m src.pipelines.prepare_for_classification -i {output_dir} -d {ssim_output_file} -o {csv2classify}
        ```

    6. Carve-outs relevance classification:

        ```python
        python -m src.pipelines.async_clf -i {csv2classify} -o {output_dir}
        ```

    7. Verification and Opportunity extraction:

        ```python
        python -m src.pipelines.finalize -i {clf_result} -o {output_dir}
        ```

3. Notebooks (more customization): `src/notebooks/`
    * Factiva API utilities: `creating_lists.ipynb`, `factiva_samples.ipynb`, `ontology.ipynb`

## Resources

[Internal Loop Doc](https://loop.cloud.microsoft/p/eyJ3Ijp7InUiOiJodHRwczovL25vcmRpY2NhcGl0YWwuc2hhcmVwb2ludC5jb20vP25hdj1jejBsTWtZbVpEMWlJWE5UZFVkSU0wZGtlbXR0VEVVeFpXcEhiamt3U0ZGQ09USTRTRkJwY2pWRWFXNU9Ra1JYVEdaVGRtVm1NR2RSU2twRVFUSlRZbGhvZUdoWU4wOWFNbFVtWmowd01WTkpSME5LUlZRelNUZFNTRkZMVlRVelRrVmFRMEpEV1VwUFJsbFZWa2hXSm1NOUptWnNkV2xrUFRFJTNEIiwiciI6ZmFsc2V9LCJwIjp7InUiOiJodHRwczovL25vcmRpY2NhcGl0YWwuc2hhcmVwb2ludC5jb20vY29udGVudHN0b3JhZ2UveDhGTk8teHRza3VDUlgyX2ZNVEhMVVJMd3hXUVYwbERvUUdnVEdNeHdZay9fbGF5b3V0cy8xNS9ndWVzdGFjY2Vzcy5hc3B4P3NoYXJlPUVRUGdEdUJ5OXo1TWs4WXNhbGVKTlk4QjlPeC1SNjZUU3ZaN3IzWkZkbmF6V0EmbmF2PWN6MGxNa1pqYjI1MFpXNTBjM1J2Y21GblpTVXlSbmc0Ums1UExYaDBjMnQxUTFKWU1sOW1UVlJJVEZWU1RIZDRWMUZXTUd4RWIxRkhaMVJIVFhoM1dXc21aRDFpSVZKVVowbEJTM2xmU3pCMWFtZEhXazFmUVRWTFpUSTNkSFl0TjJ0UlZXUkNkbVJsTldsa1NVMHRaR2RQVlV0ZmVHaERhWFpSY0VweFdIbG1aRWRyZDFBbVpqMHdNVTlQUzAxSlQxRkVORUZJVDBFMFdGaElXa2RLU0ZKU1RVNUtURmxUVGsxUUptTTlKVEpHSm1ac2RXbGtQVEVtZUQwbE4wSWxNakozSlRJeUpUTkJKVEl5VkRCU1ZGVkllSFZpTTBwcllWZE9hbGxZUW5Ca1IwWnpURzVPYjFsWVNteGpSemx3WW01UmRWa3lPWFJtUjBsb1l6Rk9NVkl3WjNwU01sSTJZVEl4VFZKVVJteGhhMlIxVDFSQ1NWVlZTVFZOYW1oSlZVZHNlVTVWVW5CaWF6VkRVa1prVFZwc1RqSmFWMWwzV2pGR1MxTnJVa0pOYkU1cFYwZG9OR0ZHWnpOVU1XOTVWbGgzZDAxV1RrcFNNRTVMVWxaUmVsTlVaRk5UUmtaTVZsUlZlbFJyVm1GUk1FcEVWMVZ3VUZKc2JGWldhMmhYSlRJeUpUSkRKVEl5YVNVeU1pVXpRU1V5TW1VMlpUaG1OemM0TFdRMU9HUXRORGsxWVMxaU5UY3lMV1V4TjJObU1qUm1OMk5tT1NVeU1pVTNSQSUzRCUzRCIsInIiOnRydWV9LCJpIjp7ImkiOiJlNmU4Zjc3OC1kNThkLTQ5NWEtYjU3Mi1lMTdjZjI0ZjdjZjkifX0%3D)  
[Factiva API examples](https://developer.dowjones.com/documents/site-docs-getting_started-postman_collections_and_python_notebooks-python_notebooks)
