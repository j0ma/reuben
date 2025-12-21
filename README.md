<h1 align="center"><pre>reuben</pre></h1>
<h3 align="center">REsampling Uncertainty Bounds for Evaluating NLP</h3>

<div align="center">
  <img src="img/reuben-banner-photo.png" width="550px" alt="REUBEN banner">
</div>

<p align="center">
  | <a href="https://arxiv.org/abs/2509.22612">Preprint @ ArXiv</a> |
  <a href="#">ACL Anthology</a> |
  <a href="https://www.jonnesaleva.com">Author website</a> |
</p>

`reuben` is a command line tool (CLI) for measuring NLP model performance
across multiple languages/tasks/test sets and quantifying uncertainty in those measurements.

It supports model comparison via pairwise differences and leaderboard-style
rankings, using most common aggregation metrics (mean, geometric mean, median).

It works by decomposing variance into components attributable to task-to-task heterogeneity ("between-variance"),
model-side randomness (eg. random seeds or draws from a model), and data-side randomness (eg. test set sampling).

## Paper

`reuben` has been featured in the paper [*Beyond statistical significance: Quantifying uncertainty and statistical variability in multilingual and multitask NLP evaluation*](https://arxiv.org/abs/2509.22612), presented at IJCNLP-AACL 2025 on December 21, 2025.

Please cite our paper as

```bibtex
@misc{sälevä2025statisticalsignificancequantifyinguncertainty,
      title={Beyond statistical significance: Quantifying uncertainty and statistical variability in multilingual and multitask NLP evaluation}, 
      author={Jonne Sälevä and Duygu Ataman and Constantine Lignos},
      year={2025},
      eprint={2509.22612},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22612}, 
}
```

## Features

- **Variance components**: What is driving performance variability?
- **Resampling**: Simulating replications to quantify uncertainty
- **Model comparison**: Which model is better, and by how much?
	- **Aggregate metrics**: arithmetic mean, geometric mean, median (±SD)  
	- **Pairwise comparisons**: (normalized) average differences
	- **Rankings**: How much uncertainty is there in model ranks?
- **Configurable**
	* **Input arguments**: YAML/JSON or CLI
	- **Output formats**: CSV or table

## Installation

First, clone the repository. Then run the following to install in "editable" mode.

```bash
pip install -e .
```

## Usage

### Input data

To run REUBEN, you need replicated outputs from each of the models you wish to compare.
The easiest way to do this is to bootstrap each test set `B` times and/or run all models model `S` on each test set you wish to include in your evaluation.

After obtaining these replicated performance scores, structure your results into a JSONL file where each line looks like this when expanded:

```json
{
  "<model column>": <model name>,
  "<task/language column>": <task name>,
  "<seed idx column>": <seed idx>,
  "<bootstrap idx column>": <bootstrap>,
  "<score column>": <score for this particular task/replication>
}
```

A concrete example might look like this:

```json
{
  "Task": "yor (MasakhaNER)"
  "Corpus": "MasakhaNER",
  "Model": "mbert-no-concat",
  "Seed": 42,
  "Bootstrap": 7,
  "F1": 72.35,
}
```


### Running the evaluation

#### Configuration

Next you will need to set up a config for your evaluation.
The preferred way to configure an analysis is using a YAML configuration file which can be passed in with `--config-file PATH`.

See below for an example of a YAML config.

#### CLI options

It is also possible to configure an analysis using CLI flags.
Run `reuben analyze --help` to see all available options for a given command.

If a flag is not provided, `reuben` will look for it in a config file (if
provided) or use a default value.

#### Example 1: Variance components analysis

The below command will estimate the variance components of each model's performance.

```bash
reuben \
	--config-file "<path to YAML config>" \
	analyze \
	--variance-components \
	--task-resampling-method none \ # <--- this overrides whatever is in the config.
	"<JSONL data file>"
```

#### Example 2: Aggregate analysis

The below command will compare each model by aggregating over tasks/languages.

```bash
reuben \
	--config-file "<path to YAML config>" \
	analyze \
	--aggregate-analysis \
	"<JSONL data file>"
```

#### Example 3: Pairwise differences

The below command will compare the models in a pairwise fashion and estimate
the variance components of the pairwise differences.

```bash
reuben \
	--config-file "<path to YAML config>" \
	 --pairwise-diffs \
	 --task-resampling-method none \
	"<JSONL data file>
```

To see an example that combines the above, see the [`example_run.sh`](example/example_run.sh) script.

### Example config file

An example config looks like this:

```yaml
# Identifiers
score_col: "score"
model_col: "model_name"
task_col: "language"

# SDs
seed_idx_col: seed_idx
boot_idx_col: boot_idx

# Resampling

## How many resamples for downstream quantities?
num_bootstrap_resamples: 10000

## How should we resample tasks in downstream eval?

task_resampling_method: "nonparametric" # "parametric" or "nonparametric"
task_resampling_with_replacement: true  # true or false
task_resampling_num_tasks: 10           # if set to less than original ==> subsampling

## Task resampling
replication_resampling_method: "nonparametric"  # "parametric" or "nonparametric"

# Output formatting

output_format: csv  # "csv" for CSV output, "rich" for table
rounding: 2
```

## Acknowledgements

- Constantine Lignos ([website](https://www.lignos.org), [lab](https://www.github.com/bltlab))
- Duygu Ataman ([website](https://www.duyguataman.com))
