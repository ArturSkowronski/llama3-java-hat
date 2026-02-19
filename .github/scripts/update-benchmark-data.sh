#!/usr/bin/env bash
# update-benchmark-data.sh
#
# Parses build/benchmark-results/results.tsv from the current run,
# merges into an existing benchmark-history.json, and produces
# a deploy-ready index.html with inlined data.
#
# Usage: ./update-benchmark-data.sh <results.tsv> <history.json> <template.html> <output-dir> [source]
#   results.tsv   - TSV from the current benchmark run
#   history.json  - existing history (or empty file / nonexistent)
#   template.html - the HTML template from docs/benchmark-page/index.html
#   output-dir    - directory to write the final index.html + data/benchmark-history.json
#   source        - run source label (default: github-actions-pocl)

set -euo pipefail

RESULTS_TSV="${1:?Usage: $0 <results.tsv> <history.json> <template.html> <output-dir>}"
HISTORY_JSON="${2:?}"
TEMPLATE_HTML="${3:?}"
OUTPUT_DIR="${4:?}"
RUN_SOURCE="${5:-github-actions-pocl}"

COMMIT_SHA="${GITHUB_SHA:-unknown}"
RUN_DATE="${RUN_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# Load existing history or start fresh
if [ -f "$HISTORY_JSON" ] && [ -s "$HISTORY_JSON" ]; then
    HISTORY=$(cat "$HISTORY_JSON")
else
    HISTORY='{"runs":[]}'
fi

# Backend name mapping from TSV names to display keys
# TSV names from InferenceBenchmarkSupport:
#   "Plain Java", "HAT Java Sequential", "HAT Java MT", "HAT OpenCL GPU"
BACKEND_KEYS=("Plain Java" "HAT Java Sequential" "HAT Java MT" "HAT OpenCL GPU")

# Parse TSV - skip header, collect only inference benchmark rows
# (not micro-benchmarks or kernel-mode rows)
declare -A RESULTS_MAP

if [ -f "$RESULTS_TSV" ]; then
    while IFS=$'\t' read -r timestamp backend load_sec infer_sec tok_per_sec error; do
        # Skip header
        [ "$timestamp" = "timestamp" ] && continue

        # Match only the 4 inference backends
        matched=""
        for key in "${BACKEND_KEYS[@]}"; do
            if [ "$backend" = "$key" ]; then
                matched="$key"
                break
            fi
        done
        [ -z "$matched" ] && continue

        # Build JSON for this result
        if [ -n "$error" ] && [ "$error" != "" ]; then
            RESULTS_MAP["$matched"]=$(jq -n \
                --arg e "$error" \
                --argjson load "$load_sec" \
                '{load_sec: $load, infer_sec: -1, tok_per_sec: -1, error: $e}')
        else
            RESULTS_MAP["$matched"]=$(jq -n \
                --argjson load "$load_sec" \
                --argjson infer "$infer_sec" \
                --argjson tok "$tok_per_sec" \
                '{load_sec: $load, infer_sec: $infer, tok_per_sec: $tok}')
        fi
    done < "$RESULTS_TSV"
fi

# Build results object
RESULTS_OBJ='{}'
for key in "${BACKEND_KEYS[@]}"; do
    if [ -n "${RESULTS_MAP[$key]+x}" ]; then
        RESULTS_OBJ=$(echo "$RESULTS_OBJ" | jq --arg k "$key" --argjson v "${RESULTS_MAP[$key]}" '. + {($k): $v}')
    else
        RESULTS_OBJ=$(echo "$RESULTS_OBJ" | jq --arg k "$key" '. + {($k): null}')
    fi
done

# Build the new run entry
NEW_RUN=$(jq -n \
    --arg date "$RUN_DATE" \
    --arg commit "$COMMIT_SHA" \
    --arg source "$RUN_SOURCE" \
    --argjson results "$RESULTS_OBJ" \
    '{date: $date, commit: $commit, source: $source, results: $results}')

# Append to history
UPDATED_HISTORY=$(echo "$HISTORY" | jq --argjson run "$NEW_RUN" '.runs += [$run]')

# Write outputs
mkdir -p "$OUTPUT_DIR/data"
echo "$UPDATED_HISTORY" | jq '.' > "$OUTPUT_DIR/data/benchmark-history.json"

# Inject data into HTML template
DATA_JSON=$(echo "$UPDATED_HISTORY" | jq -c '.')
sed "s|window.BENCHMARK_DATA = {\"runs\":\[\]};|window.BENCHMARK_DATA = ${DATA_JSON};|" \
    "$TEMPLATE_HTML" > "$OUTPUT_DIR/index.html"

echo "Benchmark page generated in $OUTPUT_DIR/"
echo "  Runs in history: $(echo "$UPDATED_HISTORY" | jq '.runs | length')"
echo "  Backends in latest run: $(echo "$RESULTS_OBJ" | jq 'keys | length')"
