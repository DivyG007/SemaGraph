# SemaGraph

Graph-vs-vector similarity experiments for Computational Linguistics 2 project P7.

## What it does

- Loads the STS Benchmark pairs from the Hugging Face `sentence-transformers/stsb` dataset.
- Parses each sentence with spaCy and converts dependency trees into NetworkX graphs.
- Scores every pair with:
	- **Graph Edit Distance similarity** (structure-sensitive)
	- **Edge-set Jaccard similarity** (lexico-syntactic overlap)
	- **Latent Semantic Analysis cosine similarity** (vector baseline)
- Compares every score against human judgments using Spearman correlation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the experiment

```bash
python code.py --split validation --sample-size 200 --ged-timeout 1.5 --log-level INFO
```

Key flags:

- `--split` – which STS split to score (`train`, `validation`, `test`).
- `--sample-size` – limit the number of pairs (omit to use the whole split).
- `--ged-timeout` – seconds the NetworkX edit-distance solver may spend per pair.
- `--lsa-components` / `--lsa-max-features` – LSA dimensionality controls.

Outputs are Spearman ρ correlations for each similarity metric versus the gold labels.