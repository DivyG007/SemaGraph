"""Standalone Latent Semantic Analysis (LSA) scoring for STS Benchmark pairs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class LSAModel:
	"""Utility wrapper that mirrors the configuration used in `code.py`."""

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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="LSA cosine similarity baseline for STS Benchmark")
	parser.add_argument("--dataset-name", default="sentence-transformers/stsb")
	parser.add_argument(
		"--split",
		default="validation",
		choices=["train", "validation", "test"],
		help="Which STS split to score",
	)
	parser.add_argument(
		"--lsa-fit-split",
		default="train",
		choices=["train", "validation", "test"],
		help="Split to train the LSA model on",
	)
	parser.add_argument("--sample-size", type=int, default=None, help="Number of pairs to evaluate")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--lsa-components", type=int, default=100)
	parser.add_argument("--lsa-max-features", type=int, default=8000)
	parser.add_argument(
		"--output",
		default="outputs/lsa_scores.parquet",
		help="Optional path to store per-pair scores (parquet)",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
	)
	return parser.parse_args()


def sample_df(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
	if sample_size is None or sample_size <= 0 or sample_size >= len(df):
		return df.copy().reset_index(drop=True)
	return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def load_sentences(dataset_name: str, split: str) -> pd.DataFrame:
	LOGGER.info("Loading %s split from %s", split, dataset_name)
	ds = load_dataset(dataset_name)
	return ds[split].to_pandas()[["sentence1", "sentence2", "score"]]


def fit_lsa(dataset_name: str, split: str, components: int, max_features: int) -> LSAModel:
	ds = load_dataset(dataset_name)
	df = ds[split].to_pandas()[["sentence1", "sentence2"]]
	sentences = list(df["sentence1"]) + list(df["sentence2"])
	model = LSAModel(components, max_features)
	model.fit(sentences)
	return model


def compute_scores(df: pd.DataFrame, embeddings: dict) -> pd.DataFrame:
	similarities: List[float] = []
	for row in tqdm(df.itertuples(index=False), total=len(df), desc="Scoring"):
		vec1 = embeddings[row.sentence1]
		vec2 = embeddings[row.sentence2]
		cos = float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0])
		similarities.append(cos)
	df = df.copy()
	df["human_score"] = df["score"] / 5.0
	df["lsa_similarity"] = similarities
	return df


def evaluate_correlations(human_scores: Sequence[float], lsa_scores: Sequence[float]) -> tuple[float, float]:
	mask = ~np.isnan(lsa_scores)
	corr, pvalue = spearmanr(np.asarray(human_scores)[mask], np.asarray(lsa_scores)[mask])
	return float(corr), float(pvalue)


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level))

	df = sample_df(load_sentences(args.dataset_name, args.split), args.sample_size, args.seed)
	unique_sentences = sorted(set(df["sentence1"]) | set(df["sentence2"]))

	model = fit_lsa(args.dataset_name, args.lsa_fit_split, args.lsa_components, args.lsa_max_features)
	embeddings = model.encode(unique_sentences)
	embedding_lookup = {sent: emb for sent, emb in zip(unique_sentences, embeddings)}

	results_df = compute_scores(df, embedding_lookup)
	corr, pvalue = evaluate_correlations(results_df["human_score"], results_df["lsa_similarity"])
	LOGGER.info("Spearman correlation with human scores: rho=%.3f (p=%.3e)", corr, pvalue)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_parquet(output_path, index=False)
	LOGGER.info("Saved per-pair scores to %s", output_path)

	print("LSA vs Human agreement: rho=%.3f (p=%.3e)" % (corr, pvalue))


if __name__ == "__main__":
	main()
