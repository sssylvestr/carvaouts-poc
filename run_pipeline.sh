#!/bin/bash

# Usage: ./run_pipeline.sh /path/to/extraction/result.csv [model_name] [existing_output_dir] [--test]
# Examples:
#   ./run_pipeline.sh extractions/job-id/result.csv                         # New run
#   ./run_pipeline.sh extractions/job-id/result.csv gpt-4.1-mini             # New run with specific model
#   ./run_pipeline.sh extractions/job-id/result.csv - processing_20250609_120717  # Continue from existing dir
#   ./run_pipeline.sh extractions/job-id/result.csv --test                  # Run in test mode with smaller datasets
#   ./run_pipeline.sh extractions/job-id/result.csv gpt-4.1-mini --test      # Run with specific model in test mode


set -e  # Exit on any error

# Parse the test flag from any position in arguments
TEST_MODE=false
for arg in "$@"; do
    if [ "$arg" == "--test" ]; then
        TEST_MODE=true
        # Remove the test flag from arguments
        set -- "${@/$arg/}"
    fi
done

# Function to check if file/directory exists and skip step if it does
function should_skip() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Output already exists: $path"
        echo "Skipping step..."
        return 0  # True - should skip
    else
        return 1  # False - should not skip
    fi
}

# Check for required argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/extraction/result.csv [model_name] [existing_output_dir] [--test]"
    echo "  model_name defaults to gpt-4.1-nano if not specified"
    echo "  Use '-' as model_name if you want to specify existing_output_dir with default model"
    echo "  Add --test to run in test mode with smaller datasets"
    exit 1
fi

# Input extraction CSV file
EXTRACTION_CSV="$1"
MODEL_NAME="${2:-gpt-4.1-nano}"

# Extract job_id from path
JOB_ID=$(echo "$EXTRACTION_CSV" | grep -o 'extractions/[^/]*/result.csv' | cut -d'/' -f2)
if [ -z "$JOB_ID" ]; then
    JOB_ID=$(dirname "$EXTRACTION_CSV" | xargs basename)
fi

# Check if continuing from existing directory
if [ $# -ge 3 ]; then
    # If third parameter is provided, use it as the output directory
    if [ "$2" == "-" ]; then
        # If second parameter is "-", use default model
        MODEL_NAME="gpt-4.1-nano"
    fi
    TIMESTAMP="$3"
    OUTPUT_DIR="extractions/${JOB_ID}/${TIMESTAMP}"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Error: Directory $OUTPUT_DIR does not exist!"
        exit 1
    fi
    
    echo "Continuing pipeline execution from existing directory: $OUTPUT_DIR"
else
    # Create new timestamped output directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="extractions/${JOB_ID}/processing_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
fi

# Create test flag for commands
TEST_FLAG=""
if [ "$TEST_MODE" = true ]; then
    TEST_FLAG="--test"
    echo "Running in TEST MODE (using smaller datasets)"
fi

echo "=== Carve-out Pipeline Started ==="
echo "Extraction file: $EXTRACTION_CSV"
echo "Job ID: $JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Test mode: $TEST_MODE"
echo "==================================="

# Setup paths for intermediate files
TILED_DIR="${OUTPUT_DIR}/tiled"
HASHED_FILE="${OUTPUT_DIR}/hashed_deduped.parquet"
SSIM_FILE="${OUTPUT_DIR}/ssim_deduped.parquet"
CSV_TO_CLASSIFY="${OUTPUT_DIR}/articles_to_classify.csv"
CLF_RESULTS_DIR="${OUTPUT_DIR}/clf_results"
FINAL_OUTPUT_DIR="${OUTPUT_DIR}/final_output"

# Create required directories if they don't exist
mkdir -p "$TILED_DIR"
mkdir -p "$CLF_RESULTS_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

# Step 1: Preprocess and tile
if ! should_skip "${TILED_DIR}/year=2024"; then
    echo "[$(date +"%H:%M:%S")] 1/6: Preprocessing and tiling articles..."
    python -m src.pipelines.preprocess_and_tile -i "$EXTRACTION_CSV" -o "$TILED_DIR" $TEST_FLAG
else
    echo "[$(date +"%H:%M:%S")] 1/6: Skipping preprocessing (output exists)"
fi

# Step 2: Hash-based deduplication
if ! should_skip "$HASHED_FILE"; then
    echo "[$(date +"%H:%M:%S")] 2/6: Hash-based deduplication..."
    python -m src.pipelines.hash_and_dedup -i "$TILED_DIR" -o "$HASHED_FILE" $TEST_FLAG
else
    echo "[$(date +"%H:%M:%S")] 2/6: Skipping hash deduplication (output exists)"
fi

# Step 3: Semantic similarity deduplication
if ! should_skip "$SSIM_FILE"; then
    echo "[$(date +"%H:%M:%S")] 3/6: Semantic similarity deduplication..."
    python -m src.pipelines.ssim_dedup -i "$HASHED_FILE" -o "$SSIM_FILE" $TEST_FLAG
else
    echo "[$(date +"%H:%M:%S")] 3/6: Skipping semantic deduplication (output exists)"
fi

# Step 4: Merge deduplication results
if ! should_skip "$CSV_TO_CLASSIFY"; then
    echo "[$(date +"%H:%M:%S")] 4/6: Merging deduplication results..."
    python -m src.pipelines.prepare_for_classification -i "$TILED_DIR" -d "$SSIM_FILE" -o "$CSV_TO_CLASSIFY" $TEST_FLAG
else
    echo "[$(date +"%H:%M:%S")] 4/6: Skipping merging dedup results (output exists)"
fi

# Step 5: Carve-outs relevance classification
CLF_RESULT=$(find "$CLF_RESULTS_DIR" -name "*.csv" 2>/dev/null | head -n 1)
if [ -z "$CLF_RESULT" ]; then
    echo "[$(date +"%H:%M:%S")] 5/6: Classifying articles for carve-out relevance..."
    python -m src.pipelines.async_clf -i "$CSV_TO_CLASSIFY" -o "$CLF_RESULTS_DIR" -m "$MODEL_NAME" $TEST_FLAG
    CLF_RESULT=$(find "$CLF_RESULTS_DIR" -name "*.csv" | head -n 1)
else
    echo "[$(date +"%H:%M:%S")] 5/6: Skipping classification (output exists)"
fi

# Step 6: Finalize and extract opportunities
FINAL_RESULT=$(find "$FINAL_OUTPUT_DIR" -name "*.xlsx" 2>/dev/null | head -n 1)
if [ -z "$FINAL_RESULT" ] && [ -n "$CLF_RESULT" ]; then
    echo "[$(date +"%H:%M:%S")] 6/6: Finalizing and extracting opportunities..."
    python -m src.pipelines.finalize -i "$CLF_RESULT" -o "$FINAL_OUTPUT_DIR" $TEST_FLAG
else
    echo "[$(date +"%H:%M:%S")] 6/6: Skipping finalization (output exists or no classification result)"
fi

echo "=== Pipeline Completed Successfully ==="
echo "Final output available at: $FINAL_OUTPUT_DIR"
echo "Pipeline execution time: $SECONDS seconds"
echo "=====================================\n"

# Print summary of files created
echo "Files created:"
find "$OUTPUT_DIR" -type f | sort

# Create a summary file
{
  echo "# Carve-out Pipeline Execution Summary"
  echo "* Date: $(date)"
  echo "* Input: $EXTRACTION_CSV"
  echo "* Model: $MODEL_NAME"
  echo "* Test mode: $TEST_MODE"
  echo "* Processing time: $SECONDS seconds"
  echo ""
  echo "## Output Locations"
  echo "* Tiled articles: $TILED_DIR"
  echo "* Hash deduplication: $HASHED_FILE"
  echo "* Semantic deduplication: $SSIM_FILE"
  echo "* Articles to classify: $CSV_TO_CLASSIFY"
  echo "* Classification results: $CLF_RESULTS_DIR"
  echo "* Final output: $FINAL_OUTPUT_DIR"
} > "${OUTPUT_DIR}/summary.md"

echo "Summary written to: ${OUTPUT_DIR}/summary.md"