"""Generate Abstract Meaning Representations (AMRs) for STS Benchmark sentences."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import amrlib
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AMR extraction helper for STS Benchmark")
	parser.add_argument("--dataset-name", default="sentence-transformers/stsb", help="Hugging Face dataset id")
	parser.add_argument(
		"--split",
		default="validation",
		choices=["train", "validation", "test"],
		help="Dataset split to parse",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=None,
		help="Limit the number of sentence pairs (default: use the whole split)",
	)
	parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
	parser.add_argument(
		"--batch-size",
		type=int,
		default=8,
		help="Number of sentences to parse per batch with the AMR model",
	)
	parser.add_argument(
		"--output",
		default="outputs/amr_pairs.jsonl",
		help="Where to store the AMR outputs (JSON Lines)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite the output file if it already exists",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
	)
	return parser.parse_args()


def load_split(dataset_name: str, split: str) -> pd.DataFrame:
	LOGGER.info("Loading %s split from %s", split, dataset_name)
	ds = load_dataset(dataset_name)
	return ds[split].to_pandas()[["sentence1", "sentence2", "score"]]


def maybe_sample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
	if sample_size is None or sample_size <= 0 or sample_size >= len(df):
		return df.copy().reset_index(drop=True)
	LOGGER.info("Sampling %d rows (seed=%d) from %d total", sample_size, seed, len(df))
	return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def unique_sentences(df: pd.DataFrame) -> List[str]:
	sents = set(df["sentence1"]) | set(df["sentence2"])
	return sorted(sents)


# Use the BART-base parse_xfm model (legacy model_stog releases removed from GitHub)
STOG_MODEL_URL = "https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz"


def _load_stog_with_fallback():
	"""Load the AMR parser, downloading weights on first run if missing."""
	try:
		return amrlib.load_stog_model()
	except FileNotFoundError:
		import amrlib.defaults as defaults
		LOGGER.info("AMR parser weights missing; downloading from %s", STOG_MODEL_URL)
		amrlib.download("model_stog", STOG_MODEL_URL, mdata_dir=defaults.data_dir)
		return amrlib.load_stog_model()


def parse_amrs(sentences: Iterable[str], batch_size: int) -> Dict[str, str]:
	LOGGER.info("Loading AMR parser (first run downloads ~1GB of weights)")
	stog = _load_stog_with_fallback()
	LOGGER.info("Parsing %d sentences in batches of %d", len(sentences), batch_size)
	amr_lookup: Dict[str, str] = {}
	batched = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]
	for batch in tqdm(batched, desc="AMR parsing"):
		graphs = stog.parse_sents(batch)
		for sent, graph in zip(batch, graphs):
			amr_lookup[sent] = graph or ""
	return amr_lookup


def build_pair_entries(df: pd.DataFrame, amr_lookup: Dict[str, str]) -> List[dict]:
	entries: List[dict] = []
	for idx, row in df.iterrows():
		entries.append(
			{
				"pair_index": int(idx),
				"sentence1": row.sentence1,
				"sentence2": row.sentence2,
				"score": float(row.score),
				"amr1": amr_lookup.get(row.sentence1, ""),
				"amr2": amr_lookup.get(row.sentence2, ""),
			}
		)
	return entries


def write_jsonl(path: Path, entries: List[dict], overwrite: bool) -> None:
	if path.exists() and not overwrite:
		raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for entry in entries:
			f.write(json.dumps(entry, ensure_ascii=False) + "\n")
	LOGGER.info("Wrote %d AMR pairs to %s", len(entries), path)


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level))

	df = maybe_sample(load_split(args.dataset_name, args.split), args.sample_size, args.seed)
	sentences = unique_sentences(df)
	amr_lookup = parse_amrs(sentences, args.batch_size)
	entries = build_pair_entries(df, amr_lookup)
	write_jsonl(Path(args.output), entries, args.overwrite)


if __name__ == "__main__":
	main()
