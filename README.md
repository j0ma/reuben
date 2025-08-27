# `reuben`

> REsampling Uncertainty Bounds for Evaluating NLP"

`reuben` is a command line tool (CLI) for measuring the NLP model performance across multiple tasks and quantifying uncertainty in those measurements.

Supports model comparison via pairwise differences and leaderboard-style rankings, using most common aggregation metrics (mean, geometric mean, median).

Decomposes variance into components attributable to task-to-task heterogeneity, model-side randomness (eg. random seeds), and data-side randomness (eg. test set sampling).

## Features

- **Variance components**: What is driving performance variability?
- **Resampling**: Simulating replications to quantify uncertainty
- **Model comparison**: Which model is better, and by how much?
	- **Aggregate metrics**: arithmetic mean, geometric mean, median (±SD)  
	- **Pairwise comparisons**: average differences, confidence intervals  
	- **Effect sizes**: scaling differences by average variability
- **Configurable**
	* **Input arguments**: YAML/JSON or CLI
	- **Output formats**: JSON/CSV or table

---

## Installation

From source:

```bash
wget -O reuben.tar.gz <url-to-tarball>
tar -xvf reuben.tar.gz

cd <release-folder-name>
pip install -e .
```
---

## Quickstart

#### Example: XQUAD

An example using the XQUAD dataset is provided in `examples/xquad`:

```
examples/xquad
├── config.yaml
└── data.jsonl
└── run.sh
```

The analysis produces a folder `examples/xquad/csv-output/` with CSV results.
Both the variance components and aggregated comparisons are run.
This is equivalent to running

```
reuben --config-file examples/config.yaml compare-models-aggregate examples/data.jsonl
reuben --config-file examples/config.yaml variance-components examples/data.jsonl
```

---


## Documentation

### Commands

#### Comparing models using aggregated scores

Compute aggregated model scores, pairwise diffs, ranks

```bash
reuben compare-models-aggregate [OPTIONS] DATA_PATH
```

Outputs:

- Aggregated results table
- Pairwise diffs per aggregator
- Effect size tables
- Rank distributions

Key options:

- Resampling  
	* `--task-resampling-method {none,nonparametric,parametric}`
	* `--task-resampling-num-tasks N`
	* `--task-resampling-with-replacement`
	* `--replication-resampling-method {none,nonparametric,parametric}`
	* `--num-bootstrap-resamples B`

- Data columns  
	* `--score-col NAME`
	* `--model-col NAME`
	* `--task-col NAME`
	* `--seed-sd-col NAME` or `--seed-idx-col NAME`
	* `--boot-sd-col NAME` or `--boot-idx-col NAME`
	* `--replication-sd-col NAME` or `--replication-idx-col NAME`

- Output options
	* `--output-format {rich,json,csv}`
	* `--output-path PATH`
	* `--rounding N`
	* `--standardized`
	* `--fix-outliers`


---

#### Variance components analysis

Summarize per-model variance components:

* Between-task SD ($\nu$)
* Within-task SD ($\eta$)
* Seed SD ($\sigma$)
* Bootstrap SD ($\tau$)
* Ratio of bootstrap SD to seed SD ($\tau / \sigma$)

Run with:

```bash
reuben variance-components [OPTIONS] DATA_PATH
```

* Options and outputs are largely the same as above.
* Outputs a table of variance components per model.

### Configuration

#### CLI options

CLI flags are the most granular way to configure `reuben`.
Run `reuben COMMAND --help` to see all available options for a given command.
If a flag is not provided, `reuben` will look for it in a config file (if provided) or use a default value.

#### Configuration file
Config files can be passed with `--config-file PATH`.

The provided [example config file](examples/xquad/config-csv-output.yaml) looks like this:

```yaml
# General
fix_outliers: true
standardized: false
output_format: csv
output_path: "./examples/xquad/csv-output"
rounding: 2

# Identifiers
score_col: "F1"
model_col: "Model"
task_col: "Language"

# Derive SDs from indices:
seed_sd_col: "SD_seed"
boot_sd_col: "SD_boot_avg"

# Resampling

## General
num_bootstrap_resamples: 100000

## Task resampling
task_resampling_method: "nonparametric"
task_resampling_with_replacement: true
task_resampling_num_tasks: 61

## Task resampling
replication_resampling_method: "parametric"
```

