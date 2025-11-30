"""Unified experiment comparing Graph, Jaccard, LSA, and Smatch similarity metrics."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import amrlib
import networkx as nx
import numpy as np
import pandas as pd
import smatch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
import spacy
from spacy.language import Language

LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def ensure_spacy_model(model_name: str) -> Language:
	try:
		return spacy.load(model_name, disable=["ner"])
	except OSError:
		LOGGER.info("spaCy model %s not found. Downloading...", model_name)
		from spacy.cli import download
		download(model_name)
		return spacy.load(model_name, disable=["ner"])


def sample_split(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
	if sample_size is None or sample_size <= 0 or sample_size >= len(df):
		return df.copy().reset_index(drop=True)
	return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Graph-based similarity
# ─────────────────────────────────────────────────────────────────────────────


def doc_to_graph(doc) -> nx.Graph:
	graph = nx.Graph()
	for token in doc:
		graph.add_node(token.i, text=token.text, lemma=token.lemma_, pos=token.pos_)
		if token.head.i != token.i:
			graph.add_edge(token.i, token.head.i, dep=token.dep_)
	return graph


def edge_signature(graph: nx.Graph) -> set[tuple[str, str, str]]:
	signatures: set[tuple[str, str, str]] = set()
	for u, v, data in graph.edges(data=True):
		lemma_u = graph.nodes[u]["lemma"]
		lemma_v = graph.nodes[v]["lemma"]
		dep = data.get("dep", "")
		first, second = sorted((lemma_u, lemma_v))
		signatures.add((first, second, dep))
	return signatures


def graph_edit_similarity(g1: nx.Graph, g2: nx.Graph, timeout: float) -> float:
	try:
		distance = nx.graph_edit_distance(g1, g2, timeout=timeout)
	except nx.NetworkXError:
		distance = None
	if distance is None or math.isinf(distance):
		return np.nan
	return 1.0 / (1.0 + distance)


def jaccard_similarity(sig_a: set, sig_b: set) -> float:
	union = len(sig_a | sig_b)
	return len(sig_a & sig_b) / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LSA similarity
# ─────────────────────────────────────────────────────────────────────────────


class LSAModel:
	def __init__(self, n_components: int, max_features: int):
		self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
		self.reducer = TruncatedSVD(n_components=n_components)
		self.normalizer = Normalizer(copy=False)
		self.pipeline = make_pipeline(self.vectorizer, self.reducer, self.normalizer)

	def fit(self, sentences: Sequence[str]) -> None:
		LOGGER.info("Fitting LSA on %d sentences", len(sentences))
		self.pipeline.fit(sentences)

	def encode(self, sentences: Sequence[str]) -> np.ndarray:
		return self.pipeline.transform(sentences)


# ─────────────────────────────────────────────────────────────────────────────
# AMR + Smatch similarity
# ─────────────────────────────────────────────────────────────────────────────


def _load_stog():
	"""Load the AMR parser (expects weights already downloaded)."""
	return amrlib.load_stog_model()


def parse_amrs(sentences: List[str], stog, batch_size: int) -> Dict[str, str]:
	LOGGER.info("Parsing %d sentences for AMR", len(sentences))
	lookup: Dict[str, str] = {}
	batched = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]
	for batch in tqdm(batched, desc="AMR parsing"):
		graphs = stog.parse_sents(batch)
		for sent, graph in zip(batch, graphs):
			lookup[sent] = graph or ""
	return lookup


def _amr_to_oneline(amr_str: str) -> str:
	"""Convert AMR string to one-line format for smatch."""
	lines = [l.strip() for l in amr_str.splitlines()]
	lines = [l for l in lines if l and not l.startswith("#")]
	oneline = " ".join(lines)
	oneline = oneline.replace("\t", " ")
	oneline = re.sub(r" +", " ", oneline)
	return oneline


def compute_smatch_score(amr1: str, amr2: str) -> float:
	"""Compute Smatch F1 between two AMR strings."""
	if not amr1 or not amr2:
		return np.nan
	try:
		one1 = _amr_to_oneline(amr1)
		one2 = _amr_to_oneline(amr2)
		smatch.match_triple_dict.clear()
		best_match, test_num, gold_num = smatch.get_amr_match(one1, one2)
		_, _, f1 = smatch.compute_f(best_match, test_num, gold_num)
		return float(f1)
	except Exception as e:
		LOGGER.warning("Smatch error: %s", e)
		return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MetricResult:
	name: str
	spearman_rho: float
	spearman_p: float
	mean_score: float
	std_score: float
	valid_count: int


@dataclass
class ExperimentResults:
	dataset_name: str
	split: str
	sample_size: int
	metrics: List[MetricResult] = field(default_factory=list)


def evaluate(args) -> ExperimentResults:
	# Load dataset
	dataset = load_dataset(args.dataset_name)
	target_df = dataset[args.split].to_pandas()[["sentence1", "sentence2", "score"]]
	sample_df = sample_split(target_df, args.sample_size, args.seed)
	LOGGER.info("Evaluating %d sentence pairs", len(sample_df))

	# Unique sentences
	unique_sentences = sorted(set(sample_df["sentence1"]) | set(sample_df["sentence2"]))

	# LSA model
	lsa_source_df = dataset[args.lsa_fit_split].to_pandas()
	lsa_sentences = list(lsa_source_df["sentence1"]) + list(lsa_source_df["sentence2"])
	lsa_model = LSAModel(args.lsa_components, args.lsa_max_features)
	lsa_model.fit(lsa_sentences)
	lsa_embeddings = lsa_model.encode(unique_sentences)
	lsa_lookup = {s: e for s, e in zip(unique_sentences, lsa_embeddings)}

	# Dependency graphs
	nlp = ensure_spacy_model(args.spacy_model)
	LOGGER.info("Parsing %d unique sentences with spaCy", len(unique_sentences))
	parsed_docs = {s: nlp(s) for s in tqdm(unique_sentences, desc="spaCy parsing")}
	graphs = {s: doc_to_graph(d) for s, d in parsed_docs.items()}
	edge_sigs = {s: edge_signature(g) for s, g in graphs.items()}

	# AMR parsing
	stog = _load_stog()
	amr_lookup = parse_amrs(unique_sentences, stog, args.amr_batch_size)

	# Score each pair
	human: List[float] = []
	ged_scores: List[float] = []
	jac_scores: List[float] = []
	lsa_scores: List[float] = []
	smatch_scores: List[float] = []

	for row in tqdm(sample_df.itertuples(index=False), total=len(sample_df), desc="Scoring pairs"):
		s1, s2 = row.sentence1, row.sentence2
		human.append(row.score / 5.0)

		ged_scores.append(graph_edit_similarity(graphs[s1], graphs[s2], timeout=args.ged_timeout))
		jac_scores.append(jaccard_similarity(edge_sigs[s1], edge_sigs[s2]))
		lsa_scores.append(float(cosine_similarity(
			lsa_lookup[s1].reshape(1, -1), lsa_lookup[s2].reshape(1, -1)
		)[0, 0]))
		smatch_scores.append(compute_smatch_score(amr_lookup[s1], amr_lookup[s2]))

	# Compute correlations
	results = ExperimentResults(
		dataset_name=args.dataset_name,
		split=args.split,
		sample_size=len(sample_df),
	)
	for name, scores in [
		("Graph Edit Distance", ged_scores),
		("Jaccard (edge-set)", jac_scores),
		("LSA cosine", lsa_scores),
		("Smatch (AMR)", smatch_scores),
	]:
		arr = np.asarray(scores)
		mask = ~np.isnan(arr)
		valid = mask.sum()
		if valid < 2:
			LOGGER.warning("Insufficient valid scores for %s", name)
			continue
		rho, p = spearmanr(np.asarray(human)[mask], arr[mask])
		results.metrics.append(MetricResult(
			name=name,
			spearman_rho=float(rho),
			spearman_p=float(p),
			mean_score=float(np.nanmean(arr)),
			std_score=float(np.nanstd(arr)),
			valid_count=int(valid),
		))

	return results


def save_results(results: ExperimentResults, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w") as f:
		json.dump({
			"dataset": results.dataset_name,
			"split": results.split,
			"sample_size": results.sample_size,
			"metrics": [
				{
					"name": m.name,
					"spearman_rho": m.spearman_rho,
					"spearman_p": m.spearman_p,
					"mean_score": m.mean_score,
					"std_score": m.std_score,
					"valid_count": m.valid_count,
				}
				for m in results.metrics
			],
		}, f, indent=2)
	LOGGER.info("Saved results to %s", output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Unified similarity metric experiment")
	parser.add_argument("--dataset-name", default="sentence-transformers/stsb")
	parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
	parser.add_argument("--lsa-fit-split", default="train")
	parser.add_argument("--sample-size", type=int, default=100)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--spacy-model", default="en_core_web_sm")
	parser.add_argument("--lsa-components", type=int, default=100)
	parser.add_argument("--lsa-max-features", type=int, default=8000)
	parser.add_argument("--ged-timeout", type=float, default=1.0)
	parser.add_argument("--amr-batch-size", type=int, default=8)
	parser.add_argument("--output", default="outputs/experiment_results.json")
	parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level))
	results = evaluate(args)

	print("\n" + "=" * 60)
	print("EXPERIMENT RESULTS")
	print("=" * 60)
	print(f"Dataset: {results.dataset_name} | Split: {results.split} | Pairs: {results.sample_size}")
	print("-" * 60)
	print(f"{'Metric':<25} {'Spearman ρ':>12} {'p-value':>12} {'Mean':>8} {'Std':>8}")
	print("-" * 60)
	for m in results.metrics:
		print(f"{m.name:<25} {m.spearman_rho:>12.4f} {m.spearman_p:>12.2e} {m.mean_score:>8.4f} {m.std_score:>8.4f}")
	print("=" * 60)

	save_results(results, Path(args.output))


if __name__ == "__main__":
	main()
