# PHONEY (a fake review classifier)

**P**inpointing **H**ollow **O**pinions **N**ot **E**arnestly **Y**ielded. ;-)

Local-first LLM classifier for fake review detection. Point it at a prompt
file and a dataset, get per-row predictions back as a CSV with accuracy and
a confusion matrix printed at the end.

Designed to run on a workstation with no internet access. Default provider
is Ollama. The provider layer is abstract; an Anthropic/OpenAI provider can
be dropped in behind the same interface.

## Getting started

### Requirements

- Python 3.14
- [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) running locally, with at least one model
  pulled (e.g. `ollama pull qwen3:14b`). The fake provider works offline
  without Ollama for plumbing checks.

### Steps

1. Install dependencies:

   ```
   poetry install
   ```

2. Download the dataset from https://osf.io/tyue9/files/3vds7 and place it at
   `data/fake-reviews-dataset.csv`. The repository does not redistribute the
   dataset.

3. Make sure the Ollama daemon is running and the model you want is pulled:

   ```
   ollama serve &           # if not already running as a service
   ollama pull qwen3:14b    # or whichever model you want as default
   ollama list              # verify it shows up
   ```

4. Run the tests to confirm everything is wired up:

   ```
   poetry run pytest
   ```

### How the Ollama connection works

The Ollama provider talks to the local daemon over HTTP. By default it
reads the `OLLAMA_HOST` environment variable if set, otherwise falls back
to `http://127.0.0.1:11434`. No network calls leave the machine.

If your daemon listens on a different host or port, set `OLLAMA_HOST`
before running:

```
OLLAMA_HOST=http://192.168.1.10:11434 poetry run phoney classify --model qwen3:14b
```

## Commands

All commands go through the `phoney` CLI.

```bash
# Classify a sample; scores are printed automatically when the run completes
poetry run phoney classify --provider ollama --model qwen3:14b --limit 17

# Run against the entire dataset
poetry run phoney classify --provider ollama --model qwen3:14b --all

# Re-run a historical prompt by its hash
poetry run phoney classify --hash 3540c00f --provider ollama --model qwen3:14b

# Score an existing results CSV (no model call)
poetry run phoney score results/qwen3_14b_3540c00f.csv

# Offline plumbing check using the fake provider (no model, no network)
poetry run phoney classify --provider fake --model fake --limit 10
```

### `classify` flags

| Flag | Description |
|------|-------------|
| `--provider` | `ollama` (default) or `fake` (offline). |
| `--model` | Model identifier passed to the provider, e.g. `qwen3:14b`. |
| `--prompt` | Path to the prompt file. Default `prompts/prompt.txt`. |
| `--hash` | 8-char prompt hash from `prompts/generations/`. Overrides `--prompt`. Fails immediately if the hash is not in the archive. |
| `--limit` | Sample size, default 200. Stratified by label. |
| `--all` | Run every row in the dataset. Overrides `--limit`. |
| `--snapshot` | Force this run into `output/` even when it would otherwise be a canonical baseline. |
| `--seed` | Sampling seed, default 42. Same seed gives the same rows. |
| `--save-prompt` | Also write every rendered prompt to a sidecar in `output/`. |
| `--verbose` | After scoring, also print a table of misclassified rows. |

### `score` flags

| Flag | Description |
|------|-------------|
| (positional) | Path to a results CSV produced by `phoney classify`. |
| `--verbose` | Also print misclassified rows. |

## Canonical runs (baselines)

A **canonical run** (also called a **baseline** or **reference baseline**)
is the current "best known" output for a particular model + prompt combination
on this repo. Its results CSV is committed to `results/` so anyone with the
repo sees the same numbers without re-running the model.

Two flavours:

| Flavour | Filename | When |
|---|---|---|
| Default sample baseline | `results/<model>_<hash>.csv` | `phoney classify` (200 stratified rows) |
| Full-dataset baseline | `results/<model>_<hash>_full.csv` | `phoney classify --all` |

Both overwrite their existing file on re-run. The baseline is always the
latest run for that shape.

### Why baselines matter

- **Reproducibility.** Months later, you can see exactly how `qwen3:14b`
  scored against prompt `3540c00f` without spinning the model up again.
- **Comparison across prompt iterations.** Edit `prompts/prompt.txt`, run
  `phoney classify` again, diff the new baseline against the old one. The
  prompt hash in the filename makes pairs obvious.
- **Regression detection.** A baseline at 78% accuracy that drops to 71% on
  re-run is a real signal: same data, same seed, same model, only the
  prompt changed.
- **Sharing.** Collaborators clone the repo and immediately have something
  scorable. They don't need GPU/CPU budget to see what the prompt does.

### How to produce a baseline

```bash
# Default 200-row baseline
poetry run phoney classify --provider ollama --model qwen3:14b

# Full 40k-row baseline (takes hours)
poetry run phoney classify --provider ollama --model qwen3:14b --all
```

Each writes to `results/<model>_<hash>[_full].csv` and commits to git like
any other change. The score is printed at the end of the run; `phoney score
<path>` re-prints it later.

### What does NOT become a baseline

These all route to `output/` (gitignored) so the canonical `results/`
directory stays clean:

- Any `--limit` that is not 200 (e.g. quick 20-row checks)
- Any run with `--snapshot` (forces output even for a default sample)
- The `--save-prompt` sidecar (always in `output/`, even when the CSV is
  canonical)

Don't move files into `results/` by hand. If a run wasn't routed there
automatically, it isn't a baseline.

### When to update a baseline

- After editing `prompts/prompt.txt`. The new prompt hash means a new
  baseline file anyway, so just re-run.
- After switching the default model.
- After a dataset change.
- **Not** after every exploratory tweak. For "I just want to see what this
  prompt does" runs, use `--snapshot` or a non-default `--limit` so you
  don't churn the baseline.

### Pairing with the prompt generations archive

A baseline CSV at `results/qwen3_14b_3540c00f.csv` always has a paired
prompt at `prompts/generations/3540c00f.txt`. Both are content-addressed by
the same hash, both are committed to git. Together they make the run fully
reproducible: the input (prompt + dataset rows by seed) and the output
(predictions) are preserved permanently.

## Reference

### Prompt generations archive

Every `classify` run snapshots the working prompt to a hash-named file. The
archive is committed, so any historical prompt can be reproduced from the
repo state:

```
prompts/
  prompt.txt                       # the working prompt; edit this
  generations/
    3540c00f.txt                   # snapshot, one per unique content hash
    f8e871f7.txt
    ...
```

Flow:

- **No `--hash`**: read `prompts/prompt.txt`, compute its 8-char SHA-256
  prefix, write `prompts/generations/<hash>.txt` if not already present.
- **With `--hash abc12345`**: load `prompts/generations/abc12345.txt` and use
  that. `prompts/prompt.txt` is ignored. Errors if the file is missing.

Each results CSV's filename embeds the same hash, so a `results/...csv` and a
`prompts/generations/<hash>.txt` always pair up.

### Result filenames and where runs land

Two directories with different lifecycles.

**`results/`: canonical baselines (committed).** One file per
`(model, prompt_hash, scope)` shape. Overwritten on each canonical run.
Anyone with the repo gets the same `results/` content for free.

**`output/`: versioned, non-canonical runs (gitignored).** Verbose
filenames so multiple runs of the same shape coexist without clobbering.
Includes the timestamp and final accuracy in the filename.

Routing rules:

| Run kind | Destination |
|---|---|
| Default (`--limit 200`), no `--snapshot` | `results/<model>_<hash>.csv` (overwrites) |
| `--all`, no `--snapshot` | `results/<model>_<hash>_full.csv` (overwrites) |
| Any other `--limit` (e.g. 50, 1000) | `output/run_<model>_<hash>_<ts>_limit_<N>_<acc>pct.csv` |
| `--snapshot` (any limit) | `output/run_..._<acc>pct.csv` |
| `--save-prompt` sidecar | Always `output/run_..._prompts.txt` regardless of CSV destination |

Why 200 is canonical: full runs on a local model take days. 200 is the
working sample size used during iteration. Treating it as the reference
shape means `phoney classify` (no flags) writes a baseline you can compare
against, every time.

Filename components:

| Component | Description |
|---|---|
| `<model>` | Model identifier with `/` and `:` replaced by `_`, e.g. `qwen3_14b`. |
| `<hash>` | First 8 hex chars of the prompt file's SHA-256. |
| `<ts>` | Run timestamp in `YYYYMMDDThhmmss` format (output filenames only). |
| `_limit_<N>` | Sample size when run was non-canonical (output only). |
| `_full` | Present when `--all` was used. |
| `_<acc>pct` | Final accuracy rounded to whole percent (output only). |

Examples:

- `results/qwen3_14b_3540c00f.csv`: canonical 200-row baseline for `qwen3:14b` + prompt `3540c00f`.
- `results/qwen3_14b_3540c00f_full.csv`: canonical full-dataset baseline.
- `output/run_qwen3_14b_3540c00f_20260520T160731_limit_50_54pct.csv`: a 50-row exploration run at 54% accuracy.

To verify a prompt hash manually:

```bash
# macOS
shasum -a 256 prompts/prompt.txt | cut -c1-8

# Linux
sha256sum prompts/prompt.txt | cut -c1-8
```

### Results CSV schema

One row per classified review, in the order they were sampled.

| Column | Description |
|---|---|
| `row_id` | Index of the row in the source `fake-reviews-dataset.csv`. Lets you join back to the original record. |
| `category` | Product category from the source row, e.g. `Home_and_Kitchen_5`. |
| `true_label` | Human ground-truth label: `CG` (computer-generated) or `OR` (original). |
| `model_label` | What the model said: `CG`, `OR`, or `UNPARSEABLE` if the response did not start with one of those two tokens. |
| `reasoning` | Short explanation extracted from the model's response (lines after the label). |
| `model_raw` | The full unmodified provider response. Useful for debugging unparseable rows. |
| `latency_ms` | Wall-clock time for that provider call, in milliseconds. |

### Save-prompt sidecar format

When `--save-prompt` is passed, every rendered prompt is written to
`output/run_<model>_<hash>_<ts>_<scope>_prompts.txt`, one entry per row, in
the same order as the CSV. Sidecars always live in `output/` even when the
CSV is canonical. The rendered prompt is reconstructable from the
generations archive and the dataset, so this is debugging context, not part
of the canonical baseline.

Each entry has a header line followed by the full prompt the model received:

```
=== row_id=14632 category=Movies_and_TV_5 true_label=OR model=qwen3:14b hash=3540c00f ===
<full rendered prompt for that row, including instruction, data block, and output contract>

=== row_id=16068 ... ===
<next prompt>
```

### Interpreting the output

A successful `classify` (or `score`) prints four sections.

**Headline.** Accuracy of answered rows, colour-coded (green ≥ 90%,
yellow ≥ 70%, red below).

**Confusion matrix.** Rows are the human label, columns are the model
label. The diagonal is correct (green). Off-diagonal cells are mistakes
(red when non-zero). A heavy off-diagonal row means the model has a bias
toward one class for that ground truth.

```
  truth ╲ pred   CG   OR
 ━━━━━━━━━━━━━━━━━━━━━━━━
  CG              3    0
  OR              1    1
```

**Classification report.** sklearn's per-class precision, recall, F1.

- *Precision*: of rows the model labelled X, how many were actually X.
- *Recall*: of rows humans labelled X, how many the model caught.
- *F1*: harmonic mean of precision and recall.
- *Support*: number of rows with that ground-truth label in the run.

**Per-category accuracy.** Match rate broken down by product category,
sorted by row count. Useful for spotting categories where the prompt
generalises poorly.

**Unparseable note.** If any model responses could not be parsed into a
label (first non-empty line was not `CG` or `OR`), a dim line is printed
below the report with the count. These rows are excluded from all metrics.
A high count usually means the prompt needs adjusting.

## Tests

`pytest` covers the dataset loader, prompt loader and generations archive,
parser, fake provider, mock-backed Ollama provider, runner end-to-end, the
scoring module, and the CLI shape.

```bash
poetry run pytest
```

Tests use stratified mini-fixtures and the `FakeProvider`, so the suite runs
in under a second and needs no network or Ollama daemon.

## Acknowledgements

Kiitos to Joni Salminen et al. for the foundational dataset.
See https://jonisalminen.com/fake-reviews-dataset-and-generation/.

## Licence

MIT. See `LICENSE`.
