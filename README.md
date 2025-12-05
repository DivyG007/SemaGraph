# SemaGraph

> Graph- and vector-based semantic similarity experiments for the Computational Linguistics 2 course project (P7).

SemaGraph benchmarks multiple sentence-similarity signals on the official Semantic Textual Similarity Benchmark (STS-B). It contrasts structural methods (dependency graphs, AMR semantics) with distributional methods (LSA, Word2Vec) inside a single, reproducible pipeline and reports how well each method aligns with human judgments.

---

## 🌟 Highlights

| Capability | Details |
| --- | --- |
| Dataset | Hugging Face `sentence-transformers/stsb` (train/validation/test) |
| Structural metrics | Graph Edit Distance, typed-edge Jaccard, AMR Smatch F1 |
| Distributional metrics | Latent Semantic Analysis (TF-IDF + Truncated SVD) and pre-trained GloVe/Word2Vec cosine |
| Statistical evaluation | Spearman ρ correlation with gold human similarity labels, descriptive stats per metric |
| Artifacts | Summary JSON, per-pair JSONL (with raw AMRs & dependency edges), publication-ready scatter plots, full LaTeX report |

---

## 📁 Repository structure

| Path | Description |
| --- | --- |
| `experiment.py` | Main experiment runner that loads STS-B, computes every metric, and emits the JSON/JSONL outputs. |
| `generate_figures.py` | Consumes per-pair logs to render correlation scatter plots for each metric (saved in `figures/`). |
| `dataset/` | Optional local cache for STS-B artifacts or custom samples (not required for Hugging Face pulling). |
| `requirements.txt` | Python dependencies, pinned to the versions used in the report. |
| `report.tex` / `report.pdf` | The course report describing the motivation, math, and findings that this code reproduces. |
| `.gitignore` | Excludes virtual environments, generated outputs, figures, and large artifacts. |

> The virtual environment itself (`.venv/`) and generated outputs (`outputs/`, `figures/`) are intentionally ignored to keep the repo lightweight.

---

## ⚙️ Setup

1. **Create a virtual environment**
	```bash
	python -m venv .venv
	source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
	```
2. **Install project dependencies**
	```bash
	pip install --upgrade pip
	pip install -r requirements.txt
	```
3. **Download language resources (first run only)**
	```bash
	python -m spacy download en_core_web_sm
	```
4. **AMR parser weights** are downloaded on demand by `amrlib`. Expect the first run to pull ~500 MB into your cache (~1 GB when including alignment models).
5. **Word embeddings** (default `glove-wiki-gigaword-100`) are auto-downloaded by `gensim.downloader` on first use.

> 📝 Tip: Keep `PYTHONPATH` clean when using Hugging Face `datasets`; the script downloads STS-B automatically and caches it under `~/.cache/huggingface/`.

---

## 🚀 Running the unified experiment

```bash
python experiment.py \
  --split validation \
  --sample-size 200 \
  --lsa-fit-split train \
  --ged-timeout 1.0 \
  --word2vec-model glove-wiki-gigaword-100 \
  --output outputs/experiment_results.json \
  --pairs-output outputs/experiment_pairs.jsonl \
  --log-level INFO
```

Key CLI flags (see `parse_args()` for the full list):

| Flag | Purpose | Default |
| --- | --- | --- |
| `--dataset-name` | Hugging Face dataset identifier | `sentence-transformers/stsb` |
| `--split` | STS-B split to score (`train`, `validation`, `test`) | `validation` |
| `--sample-size` | Number of pairs to subsample (omit for entire split) | `100` |
| `--seed` | RNG seed for reproducible sampling | `42` |
| `--lsa-components` / `--lsa-max-features` | Dimensionality controls for LSA | `100` / `8000` |
| `--ged-timeout` | Time budget per graph-edit computation | `1.0` seconds |
| `--amr-batch-size` | Sentences per AMR parsing batch | `8` |
| `--word2vec-model` | Gensim model to load | `glove-wiki-gigaword-100` |
| `--output` | Path for aggregated JSON summary | `outputs/experiment_results.json` |
| `--pairs-output` | JSON Lines dump with per-pair metrics & diagnostics | _disabled_ |

### Output files

1. **Summary JSON (`outputs/experiment_results.json`)**
	```json
	{
	  "dataset": "sentence-transformers/stsb",
	  "split": "validation",
	  "sample_size": 200,
	  "metrics": [
		 {
			"name": "Jaccard (edge-set)",
			"spearman_rho": 0.624,
			"spearman_p": 1.2e-05,
			"mean_score": 0.35,
			"std_score": 0.17,
			"valid_count": 200
		 },
		 ...
	  ]
	}
	```
2. **Per-pair JSONL (`outputs/experiment_pairs.jsonl`)** — each line stores:
	- sentence pair, raw & normalized human score
	- GED/Jaccard/LSA/Smatch/Word2Vec outputs
	- full AMR strings and dependency edge lists (handy for qualitative analysis)
3. **Figures (`figures/*.png`)** — see below for regeneration commands.

---

## 📊 Generating the correlation figures

Once you have a `pairs` JSONL file:

```bash
python generate_figures.py \
  --pairs outputs/experiment_pairs.jsonl \
  --figures-dir figures/
```

This script:
- reloads every pair record,
- filters out NaNs,
- plots metric vs. human normalized score with a least-squares trend line,
- annotates Spearman ρ/p in the title,
- saves high-resolution PNGs (300 dpi) suitable for the report.

---

## 🔬 How the pipeline works (experiment.py)

1. **Data ingestion** – Hugging Face `datasets` loads the requested split and converts it to Pandas for easier manipulation.
2. **Sampling** – `sample_split` optionally downsamples the pairs while preserving reproducibility.
3. **LSA preparation** – A large corpus (usually the train split) trains TF-IDF + TruncatedSVD; every unique sentence in the sample is encoded once.
4. **Dependency parsing** – `spaCy` parses all unique sentences; we build NetworkX graphs, typed edge signatures, and dependency edge lists.
5. **AMR parsing** – `amrlib`’s parser (stog) converts the same sentences to AMR graphs. Results are cached in a dictionary for lookup.
6. **Word2Vec encoding** – `gensim` loads the requested embeddings; sentences are lemmatized and averaged into 100-D vectors (configurable if you change the model).
7. **Scoring loop** – For every sentence pair we compute:
	- Graph Edit similarity (`graph_edit_similarity`)
	- Edge-set Jaccard similarity (`jaccard_similarity`)
	- LSA cosine (`cosine_similarity` on the encoded vectors)
	- Smatch F1 (`compute_smatch_score`)
	- Word2Vec cosine (`Word2VecModel.similarity`)
8. **Evaluation** – Using `scipy.stats.spearmanr`, we correlate each metric vector with the normalized human scores, collect mean/std/valid counts, and materialize `MetricResult` objects.
9. **Persistence** – The summary JSON and detailed JSONL are written for downstream consumption (figures, report, further research).

---

## 🧪 Reproducing the LaTeX report

The published `report.pdf` in this repo already reflects a 200-pair validation experiment. To re-run and refresh the numbers/figures:

1. Run the experiment with `--sample-size 200 --split validation --pairs-output outputs/experiment_pairs.jsonl`.
2. Generate figures (`generate_figures.py`).
3. Update tables/plots inside `report.tex` (the file already references the PNGs in `figures/`).
4. Compile the LaTeX report:
	```bash
	latexmk -pdf report.tex
	```

---

## 🧰 Troubleshooting & tips

- **AMR parsing is slow** – reduce `--sample-size` or `--amr-batch-size`; AMR is the dominant cost.
- **Graph Edit distance fails/timeouts** – increase `--ged-timeout` or skip GED by ignoring that metric when post-processing.
- **Missing spaCy model** – rerun `python -m spacy download en_core_web_sm`.
- **Word embedding download stalls** – prefetch using `python -c "import gensim.downloader as api; api.load('glove-wiki-gigaword-100')"`.
- **Large virtual env** – keep `.venv/` untracked; only requirements are version-pinned.
- **Determinism** – set `--seed` and avoid multithreading in dependencies if bitwise determinism is needed.

---

## 🤝 Contributing / extending

1. Fork the repo, create a new branch, and keep heavy artifacts out of Git (use `.gitignore` as a guide).
2. For new metrics, plug into the scoring loop inside `evaluate()` and append to the `metrics` list for evaluation.
3. Update this README and `report.tex` if you change the experimental design.

Happy graphing! 🎯