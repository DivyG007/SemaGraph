#!/usr/bin/env python3
"""Generate scatter plots for metric vs human STS scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def load_pair_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def scatter_metric(records: List[Dict], metric_key: str, title: str, out_path: Path) -> None:
    humans: List[float] = []
    metric_values: List[float] = []
    for rec in records:
        value = rec.get(metric_key)
        human = rec.get("human_score_norm")
        if value is None or human is None:
            continue
        if np.isnan(value) or np.isnan(human):
            continue
        humans.append(float(human))
        metric_values.append(float(value))

    if not humans:
        raise ValueError(f"No valid pairs for metric {metric_key}")

    rho, p = spearmanr(humans, metric_values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.2))
    plt.scatter(humans, metric_values, alpha=0.6, edgecolor="none")

    xs = np.linspace(min(humans), max(humans), 100)
    z = np.polyfit(humans, metric_values, 1)
    ys = np.poly1d(z)(xs)
    plt.plot(xs, ys, color="tab:red", linestyle="--", linewidth=1.5, label="Least-squares fit")

    plt.xlabel("Human STS score (0-1)")
    plt.ylabel(title)
    plt.title(f"{title} vs Human (ρ={rho:.3f}, p={p:.1e})")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scatter plots for experiment metrics")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/experiment_pairs_w2v.jsonl"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    records = load_pair_records(args.pairs)
    metrics = {
        "ged_similarity": "Graph Edit Similarity",
        "jaccard_similarity": "Jaccard Edge Overlap",
        "lsa_cosine": "LSA Cosine",
        "smatch_f1": "Smatch F1",
        "word2vec_cosine": "Word2Vec Cosine",
    }
    for key, label in metrics.items():
        out_path = args.figures_dir / f"{key}_vs_human.png"
        scatter_metric(records, key, label, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
