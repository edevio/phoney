# fake-review-classifier

Local-first LLM classifier for fake review detection. Point it at a prompt
file and a dataset, get per-row predictions back as a CSV.

Designed to run on a workstation with no internet access. Default provider is
Ollama. 

## Install

Requires Python 3.14 and [Poetry](https://python-poetry.org/).

```
poetry install
```

## Dataset

Place `fake-reviews-dataset.csv` in `data/`. The repository does not
redistribute the dataset. (Joni Salimen et al. data used - [See here](https://jonisalminen.com/fake-reviews-dataset-and-generation/).

## Ackowledgements

Kiitos to Joni Salminen et al for the foundational dataset.

## Licence

MIT. See `LICENSE`.
