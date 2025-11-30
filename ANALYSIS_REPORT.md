# SemaGraph: Comparative Analysis of Sentence Similarity Metrics

**Computational Linguistics 2 – Project P7**

---

## 1. Introduction

This report presents a comprehensive evaluation of four sentence similarity metrics on the STS Benchmark dataset. The goal is to compare **graph-based** and **vector-based** approaches for capturing semantic similarity, with a particular focus on how Abstract Meaning Representation (AMR) and Smatch scoring perform relative to traditional baselines.

### Metrics Evaluated

| Metric | Type | Description |
|--------|------|-------------|
| **Graph Edit Distance (GED)** | Graph-based | Measures structural similarity between dependency parse trees |
| **Jaccard (edge-set)** | Graph-based | Computes overlap of lemma-labeled dependency edges |
| **LSA cosine** | Vector-based | TF-IDF + Truncated SVD embeddings with cosine similarity |
| **Smatch (AMR)** | Semantic graph | F1-score between Abstract Meaning Representations |

---

## 2. Dataset

### STS Benchmark

The **Semantic Textual Similarity Benchmark** (STS-B) is a standard evaluation dataset for sentence similarity models. It contains sentence pairs with human-annotated similarity scores on a 0–5 scale.

| Split | Pairs | Purpose |
|-------|-------|---------|
| Train | 5,749 | LSA model fitting |
| Validation | 1,500 | Primary evaluation |
| Test | 1,379 | Held-out evaluation |

**Source:** `sentence-transformers/stsb` on Hugging Face Datasets

### Sample Statistics (Validation Split, n=200)

- **Mean human score:** 2.96 (normalized: 0.59)
- **Score distribution:** Covers full 0–5 range with slight positive skew
- **Sentence length:** Average ~12 tokens per sentence

---

## 3. Methodology

### 3.1 Dependency Graph Construction

Each sentence is parsed using **spaCy** (`en_core_web_sm`). The dependency tree is converted to an undirected NetworkX graph where:
- **Nodes** represent tokens (with lemma, POS attributes)
- **Edges** represent dependency relations (with dependency type labels)

### 3.2 Graph Edit Distance

Graph Edit Distance (GED) computes the minimum cost to transform one graph into another through node/edge insertions, deletions, and substitutions. We convert this to a similarity score:

$$\text{GED\_sim}(G_1, G_2) = \frac{1}{1 + \text{GED}(G_1, G_2)}$$

**Timeout:** 1.0 second per pair (NP-hard computation)

### 3.3 Jaccard Edge-Set Similarity

Extracts edge signatures as `(lemma_a, lemma_b, dep_type)` tuples and computes:

$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### 3.4 Latent Semantic Analysis (LSA)

1. Build TF-IDF vectors with unigrams and bigrams (max 8,000 features)
2. Apply Truncated SVD to reduce to 100 dimensions
3. L2-normalize embeddings
4. Compute cosine similarity between sentence vectors

**Training corpus:** All sentences from the train split (~11,500 sentences)

### 3.5 Abstract Meaning Representation + Smatch

1. Parse each sentence to AMR using **amrlib** (BART-base model, SMATCH ≈ 82.3)
2. Compute Smatch F1 between AMR graphs:

$$\text{Smatch} = \frac{2 \cdot P \cdot R}{P + R}$$

Where precision (P) and recall (R) measure triple overlap after optimal variable alignment.

---

## 4. Results

### 4.1 Correlation with Human Judgments

| Metric | Spearman ρ | p-value | Interpretation |
|--------|------------|---------|----------------|
| **Smatch (AMR)** | **0.618** | 1.97e-22 | Strong positive correlation |
| **Jaccard (edge-set)** | **0.624** | 5.14e-23 | Strong positive correlation |
| **LSA cosine** | 0.546 | 6.51e-17 | Moderate positive correlation |
| Graph Edit Distance | 0.010 | 8.93e-01 | No correlation |

### 4.2 Score Distributions

| Metric | Mean | Std Dev | Valid Pairs |
|--------|------|---------|-------------|
| Graph Edit Distance | 0.213 | 0.281 | 200 |
| Jaccard (edge-set) | 0.180 | 0.201 | 200 |
| LSA cosine | 0.550 | 0.291 | 200 |
| Smatch (AMR) | 0.502 | 0.216 | 200 |

---

## 5. Analysis

### 5.1 Key Findings

1. **Smatch and Jaccard perform comparably** (ρ ≈ 0.62), both outperforming LSA. This suggests that **structural/semantic graph features** capture aspects of similarity that distributional vectors miss.

2. **Graph Edit Distance fails** on this task (ρ ≈ 0). The metric is too sensitive to minor structural differences and struggles with variable sentence lengths. The NP-hard computation also leads to timeouts on complex graphs.

3. **LSA provides a solid baseline** (ρ = 0.55) with minimal computational cost. It captures lexical overlap and topical similarity but misses fine-grained semantic distinctions.

4. **AMR abstracts away surface variation** better than dependency trees. Smatch's strong performance validates the utility of meaning representations for similarity tasks.

### 5.2 Error Analysis

**Cases where Smatch excels over LSA:**
- Paraphrases with different lexical choices but same meaning
- Active/passive voice alternations
- Sentences with shared predicates but different modifiers

**Cases where Smatch struggles:**
- Parser errors propagate to similarity scores
- Named entities parsed inconsistently
- Very short sentences (minimal graph structure)

### 5.3 Computational Considerations

| Metric | Parsing Time | Scoring Time | Total (200 pairs) |
|--------|--------------|--------------|-------------------|
| GED | ~1s (spaCy) | ~2 min (timeouts) | ~3 min |
| Jaccard | ~1s (spaCy) | <1s | ~1s |
| LSA | N/A (batch fit) | <1s | ~2s |
| Smatch | ~60s (AMR) | ~2s | ~65s |

AMR parsing is the bottleneck for Smatch. For large-scale deployment, consider caching AMR parses or using faster models.

---

## 6. Conclusions

1. **Semantic graph metrics (Smatch, Jaccard) outperform pure vector baselines** on STS Benchmark, achieving ρ > 0.6 vs human judgments.

2. **Graph Edit Distance is unsuitable** for sentence similarity due to computational complexity and over-sensitivity to structural noise.

3. **AMR-based Smatch scoring** offers the best balance of semantic fidelity and interpretability, though at higher computational cost than surface-level methods.

4. **Hybrid approaches** combining vector embeddings (for efficiency) with graph features (for precision) are a promising direction for future work.

---

## 7. Reproducibility

### Running the Experiments

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run unified experiment
python experiment.py --sample-size 200 --split validation --output outputs/results.json

# View results
cat outputs/results.json | python -m json.tool
```

### Files

| File | Purpose |
|------|---------|
| `experiment.py` | Unified evaluation script (all 4 metrics) |
| `main.py` | Original comparison script (GED, Jaccard, LSA) |
| `amr_pipeline.py` | Standalone AMR extraction to JSONL |
| `lsa_analysis.py` | Standalone LSA baseline |

---

## References

1. Cer, D., et al. (2017). SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation.
2. Banarescu, L., et al. (2013). Abstract Meaning Representation for Sembanking.
3. Cai, S., & Knight, K. (2013). Smatch: An Evaluation Metric for Semantic Feature Structures.
4. Deerwester, S., et al. (1990). Indexing by Latent Semantic Analysis.
