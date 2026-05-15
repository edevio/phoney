# fake-review-classifier

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

Run a classification pass against a sample using the offline fake provider:

```
poetry run phoney run --provider fake --model fake --limit 200 --seed 42
```

Results land in `results/<model>_<prompt-hash>.csv` with one row per
classified review.

## What works so far

- Dataset loader: reads the CSV into typed `Review` records, with optional
  stratified sampling by label and a deterministic `--seed`.
- `phoney preview`: prints a Rich table of sampled rows.
- Prompt loader: reads a prompt file, computes a stable 8-char hash, and
  builds the matching `results/<model>_<hash>.csv` path.
- Provider abstraction with an offline `FakeProvider` for tests and plumbing.
  Real providers (Ollama, etc.) drop in behind the same interface.
- `phoney run`: iterates a sample, classifies each review, parses the
  response into a label and reasoning, writes a results CSV, shows a live
  Rich progress bar. End-to-end works offline with the fake provider.

## Acknowledgements

Kiitos to Joni Salminen et al. for the foundational dataset.
See https://jonisalminen.com/fake-reviews-dataset-and-generation/.

## Licence

MIT. See `LICENSE`.
