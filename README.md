# SemaGraph

Graph-vs-vector similarity experiments for Computational Linguistics 2 project P7.

## What it does

- Loads the STS Benchmark pairs from the Hugging Face `sentence-transformers/stsb` dataset.
- Parses each sentence with spaCy and converts dependency trees into NetworkX graphs.
- Parses each sentence to Abstract Meaning Representation (AMR) using amrlib.
- Scores every pair with:
	- **Graph Edit Distance similarity** (structure-sensitive)
	- **Edge-set Jaccard similarity** (lexico-syntactic overlap)
	- **Latent Semantic Analysis cosine similarity** (vector baseline)
	- **Smatch F1** (AMR semantic graph similarity)
- Compares every score against human judgments using Spearman correlation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Note:** The first run of AMR parsing downloads ~500 MB of model weights. These are cached for future runs.

## Run the unified experiment (all metrics)

```bash
python experiment.py --split validation --sample-size 200 --ged-timeout 1.0 --log-level INFO
```

This runs Graph Edit Distance, Jaccard, LSA, and Smatch on the same dataset sample and outputs Spearman correlations.

## Run the original experiment (without Smatch)

```bash
python main.py --split validation --sample-size 200 --ged-timeout 1.5 --log-level INFO
```

Key flags:

- `--split` – which STS split to score (`train`, `validation`, `test`).
- `--sample-size` – limit the number of pairs (omit to use the whole split).
- `--ged-timeout` – seconds the NetworkX edit-distance solver may spend per pair.
- `--lsa-components` / `--lsa-max-features` – LSA dimensionality controls.

Outputs are Spearman ρ correlations for each similarity metric versus the gold labels.

## AMR extraction only

Use `amr_pipeline.py` when you just need Abstract Meaning Representations for each sentence in an STS split. The script caches all unique sentences, parses them with the pretrained AMR parser from `amrlib`, and stores the results as JSON Lines (one record per sentence pair).

```bash
python amr_pipeline.py --split validation --sample-size 50 --batch-size 4 --output outputs/amr_val.jsonl
```

> ℹ️ The first run downloads ~1 GB of AMR weights. Re-use the cached model afterwards.

## Standalone LSA baseline

`lsa_analysis.py` mirrors the LSA baseline inside `code.py`, but focuses solely on TF-IDF + truncated SVD comparisons. It writes every scored pair (including the normalized human label) to Parquet so you can inspect or plot the results separately.

```bash
python lsa_analysis.py --split validation --sample-size 200 --lsa-components 120 --lsa-max-features 10000 --output outputs/lsa_validation.parquet
```

The command prints the Spearman correlation against the human scores and saves the detailed per-pair cosine similarities for downstream analysis.