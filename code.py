"""Entry point for comparing graph-based and LSA-based sentence similarities."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from datasets import load_dataset
import spacy
from scipy.stats import spearmanr
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from spacy.language import Language

LOGGER = logging.getLogger(__name__)


def ensure_spacy_model(model_name: str) -> Language:
	"""Load (and download if necessary) a spaCy model."""

	try:
		return spacy.load(model_name, disable=["ner"])
	except OSError:
		LOGGER.info("spaCy model %s not found. Downloading...", model_name)
		from spacy.cli import download

		download(model_name)
		return spacy.load(model_name, disable=["ner"])


def load_sts_dataset(dataset_name: str):
	"""Load the Sentence-Transformers STS Benchmark dataset."""

	LOGGER.info("Loading dataset %s", dataset_name)
	return load_dataset(dataset_name)


def sample_split(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
	if sample_size is None or sample_size <= 0 or sample_size >= len(df):
		return df.copy().reset_index(drop=True)
	return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def doc_to_graph(doc) -> nx.Graph:
	graph = nx.Graph()
	for token in doc:
		graph.add_node(
			token.i,
			text=token.text,
			lemma=token.lemma_,
			pos=token.pos_,
		)
		if token.head.i == token.i:
			continue
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
	if union == 0:
		return 0.0
	return len(sig_a & sig_b) / union


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


@dataclass
class SimilarityResult:
	name: str
	spearman: float
	pvalue: float


def evaluate(args) -> List[SimilarityResult]:
	dataset = load_sts_dataset(args.dataset_name)
	target_df = dataset[args.split].to_pandas()[["sentence1", "sentence2", "score"]]
	sample_df = sample_split(target_df, args.sample_size, args.seed)

	lsa_source_df = dataset[args.lsa_fit_split].to_pandas()
	lsa_sentences = list(lsa_source_df["sentence1"]) + list(lsa_source_df["sentence2"])
	lsa_model = LSAModel(args.lsa_components, args.lsa_max_features)
	lsa_model.fit(lsa_sentences)

	unique_sentences = sorted(set(sample_df["sentence1"]) | set(sample_df["sentence2"]))
	nlp = ensure_spacy_model(args.spacy_model)
	LOGGER.info("Parsing %d unique sentences", len(unique_sentences))
	parsed_docs = {sent: nlp(sent) for sent in tqdm(unique_sentences, desc="Parsing")}

	graphs = {sent: doc_to_graph(doc) for sent, doc in parsed_docs.items()}
	edge_signatures = {sent: edge_signature(graph) for sent, graph in graphs.items()}

	embeddings = lsa_model.encode(unique_sentences)
	embedding_lookup = {sent: emb for sent, emb in zip(unique_sentences, embeddings)}

	human_scores: List[float] = []
	graph_edit_scores: List[float] = []
	jaccard_scores: List[float] = []
	lsa_scores: List[float] = []

	for row in tqdm(sample_df.itertuples(index=False), total=len(sample_df), desc="Scoring"):
		sent1 = row.sentence1
		sent2 = row.sentence2

		g1 = graphs[sent1]
		g2 = graphs[sent2]
		sig1 = edge_signatures[sent1]
		sig2 = edge_signatures[sent2]

		ged_sim = graph_edit_similarity(g1, g2, timeout=args.ged_timeout)
		jac_sim = jaccard_similarity(sig1, sig2)
		lsa_sim = float(
			cosine_similarity(
				embedding_lookup[sent1].reshape(1, -1),
				embedding_lookup[sent2].reshape(1, -1),
			)[0, 0]
		)

		human_scores.append(row.score / 5.0)
		graph_edit_scores.append(ged_sim)
		jaccard_scores.append(jac_sim)
		lsa_scores.append(lsa_sim)

	return compute_correlations(human_scores, graph_edit_scores, jaccard_scores, lsa_scores)


def compute_correlations(
	human: Sequence[float],
	graph_edit: Sequence[float],
	jaccard: Sequence[float],
	lsa: Sequence[float],
) -> List[SimilarityResult]:
	results = []
	pairs = {
		"graph_edit": graph_edit,
		"jaccard": jaccard,
		"lsa": lsa,
	}

	human_arr = np.asarray(human)
	for name, scores in pairs.items():
		score_arr = np.asarray(scores)
		mask = ~np.isnan(score_arr)
		if mask.sum() < 2:
			LOGGER.warning("Insufficient valid scores for %s", name)
			continue
		corr, pvalue = spearmanr(human_arr[mask], score_arr[mask])
		results.append(SimilarityResult(name=name, spearman=float(corr), pvalue=float(pvalue)))
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Graph vs LSA sentence similarity evaluator")
	parser.add_argument("--dataset-name", default="sentence-transformers/stsb")
	parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
	parser.add_argument("--lsa-fit-split", default="train", choices=["train", "validation", "test"])
	parser.add_argument("--sample-size", type=int, default=200, help="Number of sentence pairs to score")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--spacy-model", default="en_core_web_sm")
	parser.add_argument("--lsa-components", type=int, default=100)
	parser.add_argument("--lsa-max-features", type=int, default=8000)
	parser.add_argument("--ged-timeout", type=float, default=1.0, help="Timeout (seconds) for graph edit distance")
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level))
	results = evaluate(args)
	print("\n--- Similarity vs Human Judgments (Spearman) ---")
	for result in results:
		print(f"{result.name:>12}: rho={result.spearman:.3f} (p={result.pvalue:.3e})")


if __name__ == "__main__":
	main()

