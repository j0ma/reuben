#!/usr/bin/env bash
set -euo pipefail

# Usage: ./script.sh <metric> <root_folder>

run_reuben_pipeline() {
  local metric="$1"
  local root="${2}"

  local data="$root/data.jsonl"
  local config="$root/config-${metric}-csv-output.yaml"
  local outdir="$root/csv-output/${metric}"

  mkdir -p "$outdir"

  rich -p '[green](1/3) Aggregate analysis... [/green]' #2>/dev/null
  reuben --config-file "$config" analyze --aggregate-analysis "$data" --output-path "$outdir" #2>/dev/null

  rich -p '[green](2/3) Variance components... [/green]' #2>/dev/null
  reuben --config-file "$config" analyze --variance-components --task-resampling-method none "$data" --output-path "$outdir" #2>/dev/null

  rich -p '[green](3/3) Pairwise difference variance components... [/green]' #2>/dev/null
  reuben --config-file "$config" analyze --pairwise-diffs --pairwise-diffs-task-level-details --task-resampling-method none "$data" --output-path "$outdir" #2>/dev/null

  rich -p '[green]🎉 Done! 🎉[/green]' 2>/dev/null
  rich -p "[green]Output located in $outdir [/green]" 2>/dev/null
  tree "$outdir"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <metric> [root]"
  exit 1
fi

script_folder=$(dirname $BASH_SOURCE)
run_reuben_pipeline "$1" "${2:-$script_folder}"
