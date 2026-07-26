#!/usr/bin/env python3

"""Run Python openTSNE from precomputed neighbors and a fixed initialization."""

import argparse
import csv
import os
import time

import numpy as np
from openTSNE import TSNEEmbedding
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.nearest_neighbors import PrecomputedNeighbors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True)
    parser.add_argument("--distances", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--output-layout", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--perplexity", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--early-iterations", default=250, type=int)
    parser.add_argument("--normal-iterations", default=750, type=int)
    parser.add_argument(
        "--negative-gradient-method",
        choices=("fft", "exact"),
        default="fft",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    indices = np.fromfile(args.indices, dtype=np.int32).reshape(args.n, args.k)
    distances = np.fromfile(args.distances, dtype=np.float32).reshape(args.n, args.k)
    initialization = np.fromfile(
        args.initialization, dtype=np.float32
    ).reshape(args.n, 2)

    start = time.perf_counter()
    neighbors = PrecomputedNeighbors(indices, distances)
    affinities = PerplexityBasedNN(
        data=None,
        perplexity=args.perplexity,
        knn_index=neighbors,
        symmetrize=True,
        n_jobs=args.threads,
        random_state=args.seed,
        verbose=False,
    )
    affinity_sec = time.perf_counter() - start

    learning_rate = args.n / 12.0
    reference_method = (
        "fft" if args.negative_gradient_method == "fft" else "bh"
    )
    embedding = TSNEEmbedding(
        initialization.copy(),
        affinities,
        negative_gradient_method=reference_method,
        random_state=args.seed,
        n_jobs=args.threads,
        learning_rate=learning_rate,
        momentum=0.8,
        max_step_norm=5.0,
        theta=0.0 if args.negative_gradient_method == "exact" else 0.5,
        verbose=False,
    )
    optimize_start = time.perf_counter()
    embedding.optimize(
        args.early_iterations,
        inplace=True,
        exaggeration=12.0,
        learning_rate=learning_rate,
        momentum=0.8,
        max_step_norm=5.0,
    )
    embedding.optimize(
        args.normal_iterations,
        inplace=True,
        exaggeration=1.0,
        learning_rate=learning_rate,
        momentum=0.8,
        max_step_norm=5.0,
    )
    optimization_sec = time.perf_counter() - optimize_start
    total_sec = time.perf_counter() - start

    layout = np.asarray(embedding, dtype=np.float32, order="C")
    layout.tofile(args.output_layout)
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    with open(args.output_metrics, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "implementation",
                "backend",
                "seed",
                "n",
                "k",
                "perplexity",
                "affinity_sec",
                "optimization_sec",
                "total_sec",
                "reported_kl",
                "negative_gradient_method",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "implementation": "Python openTSNE",
                "backend": "python_cpu",
                "seed": args.seed,
                "n": args.n,
                "k": args.k,
                "perplexity": args.perplexity,
                "affinity_sec": affinity_sec,
                "optimization_sec": optimization_sec,
                "total_sec": total_sec,
                "reported_kl": float(embedding.kl_divergence),
                "negative_gradient_method": args.negative_gradient_method,
            }
        )


if __name__ == "__main__":
    main()
