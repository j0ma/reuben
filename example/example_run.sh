#!/usr/bin/env bash
set -euo pipefail

# If example/example_data.jsonl doesn't exist download it
if [ ! -f "example/example_data.jsonl" ]; then
  rich -p '[green]Downloading data... [/green]'
  curl -L -o example/example_data.jsonl https://cs.brandeis.edu/~jonne/reuben/example_data.jsonl
fi

run_reuben_pipeline() {

  local data="example/example_data.jsonl"
  local config="example/config-seed-boot-richtable.yaml"

  (
    rich -p '[green](1/3) Aggregate analysis... [/green]'
    reuben --config-file "$config" analyze --aggregate-analysis "$data"

    rich -p '[green](2/3) Variance components... [/green]' 
    reuben --config-file "$config" analyze --variance-components --task-resampling-method none "$data" 

    rich -p '[green](3/3) Pairwise difference variance components... [/green]'
    reuben --config-file "$config" analyze --pairwise-diffs --task-resampling-method none "$data" 

    rich -p '[green]🎉 Done! 🎉[/green]' 
  ) 2> /dev/null
}

run_reuben_pipeline
