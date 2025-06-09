# Carve-outs POC

PoC is dedicated to detecting companies that may be potentially interested in doing a corporate carve-out.
Wider case is covering building a separate pipeline that would aggregate and process articles from different sources and detect various news signals that may be of interest for Investment Professionals.

## External Sources used

* Factiva News

TBA:

* Factset Transcripts (API/DataBricks)
* Mergermarket (M&A)

## Factiva extraction pipeline

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
Using UV:

```bash
uv venv  
uv sync --extra_index_url $PIP_EXTRA_INDEX_URL #basic installation
uv sync --extra_index_url $PIP_EXTRA_INDEX_URL --all-extras #extras: semantic similarity; jupyter
```

2. Running the pipeline:
    * Form extraction query and execute it; download data (be very careful due to quotas! - better to run it in a notebook ): `src/notebooks/data_extraction.ipynb`
        Input: None
        Output: extractions/{job_id}/result.csv
    * Preprocess and tile articles into smaller ones (optimized): `python -m src.pipelines.preprocess_and_tile -i {extraction_csv} -o {output_dir}`
        Input: extractions/{job_id}/result.csv
        Output: {output_dir} with a set of .parquet files
    * Deduplicate using non-semantic hashing methods (not optimized): `python -m src.pipelines.hash_and_dedup -i {output_dir} -o {output_file}`
        Input: {output_dir} with a set of .parquet files
    * Semantic deduplication (not optimized): `python -m src.pipelines.ssim_dedup -i {output_file} -o {ssim_output_file}`
    * Merging deduplicated article segments with useful metadata: `python -m src.pipelines.prepare_for_classification -i {output_dir} -d {ssim_output_file} -o {csv2classify}`
    * Carve-outs relevance classification: `python -m src.pipelines.async_clf -i {csv2classify} -o {output_dir}`
    * Verification and Opportunity extraction: `python -m src.pipelines.finalize -i {clf_result} -o {output_dir}`

3. Notebooks (more customization possiblities): `src/notebooks/`
    * Factiva API utilities: `creating_lists.ipynb`, `factiva_samples.ipynb`, `ontology.ipynb`

## Useful materials

[Internal Loop Doc](TBA)  
[Factiva API examples](https://developer.dowjones.com/documents/site-docs-getting_started-postman_collections_and_python_notebooks-python_notebooks)
