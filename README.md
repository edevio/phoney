# PHONEY (a fake review classifier)

**P**inpointing **H**ollow **O**pinions **N**ot **E**arnestly **Y**ielded. ;-) 

Local-first LLM classifier for fake review detection. Point it at a prompt
file and a dataset, get per-row predictions back as a CSV.

Designed to run on a workstation with no internet access. Default provider is
Ollama.

## Getting started

Requires Python 3.14 and [Poetry](https://python-poetry.org/).

1. Install dependencies:

   ```
   poetry install
   ```

2. Download the dataset from https://osf.io/tyue9/files/3vds7 and place it at
   `data/fake-reviews-dataset.csv`. The repository does not redistribute the
   dataset.

3. Run the tests to confirm everything is wired up:

   ```
   poetry run pytest
   ```

## Usage

Preview a sample of rows from the dataset:

```
poetry run phoney preview --limit 5 --seed 42
```

Classify a sample of reviews. Scores are printed automatically when the run
completes. Default is a local Ollama model:

```
poetry run phoney classify --provider ollama --model qwen3:14b --limit 200 --seed 42
```

Or run against the entire dataset:

```
poetry run phoney classify --provider ollama --model qwen3:14b --all
```

Or use the offline fake provider for plumbing checks:

```
poetry run phoney classify --provider fake --model fake --limit 200 --seed 42
```

Results land in `results/<model>_<prompt-hash>.csv` with one row per
classified review.

Pass `--save-prompt` to also write every rendered prompt to a sidecar at
`results/<model>_<prompt-hash>_prompts.txt`. Each entry has a header line
naming the row, followed by the full prompt the model received for that row.

Score an existing results CSV without re-running the model:

```
poetry run phoney score results/qwen3_14b_b0cef827.csv
```

Add `--verbose` to also see the misclassified rows.

## What works so far

- Dataset loader: reads the CSV into typed `Review` records, with optional
  stratified sampling by label and a deterministic `--seed`.
- `phoney preview`: prints a Rich table of sampled rows.
- Prompt loader: reads a prompt file, computes a stable 8-char hash, and
  builds the matching `results/<model>_<hash>.csv` path.
- Provider abstraction with an offline `FakeProvider` for tests and plumbing,
  and an `OllamaProvider` for local models via the Ollama daemon.
- `phoney classify`: iterates a sample, classifies each review, parses the
  response into a label and reasoning, writes a results CSV, shows a live
  Rich progress bar, and prints accuracy/confusion/per-category at the end.
  `--verbose` also prints misclassified rows.
- `--save-prompt` writes every rendered prompt (one per row, with headers)
  to a sidecar next to the CSV.
- `--all` runs against every row in the dataset and overrides `--limit`.
- `phoney score <results.csv>`: same scoring view applied to an existing
  results CSV. `--verbose` adds the misclassified rows table. Unparseable
  rows are excluded from scoring; a note is printed below the report if any
  were present.

## Acknowledgements

Kiitos to Joni Salminen et al. for the foundational dataset.
See https://jonisalminen.com/fake-reviews-dataset-and-generation/.

## Licence

MIT. See `LICENSE`.
