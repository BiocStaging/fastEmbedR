#!/usr/bin/env python3

"""Build an exact Python openTSNE affinity matrix for validation only."""

import argparse
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--p", required=True, type=int)
    parser.add_argument("--perplexity", required=True, type=float)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(args.output), "numba_cache"))
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

    from openTSNE import affinity

    x = np.fromfile(args.input, dtype="<f4")
    expected = args.n * args.p
    if x.size != expected:
        raise RuntimeError(f"Expected {expected} float32 values, found {x.size}")
    x = np.ascontiguousarray(x.reshape(args.n, args.p))
    model = affinity.PerplexityBasedNN(
        x,
        perplexity=min(args.perplexity, max(1, (args.n - 1) / 3)),
        method="exact",
        metric="euclidean",
        n_jobs=args.threads,
        random_state=args.seed,
        k_neighbors=min(args.k, args.n - 1),
        verbose=False,
    )
    p_matrix = model.P.tocoo()
    with open(args.output, "wb") as handle:
        np.asarray([p_matrix.nnz], dtype="<i4").tofile(handle)
        np.asarray(p_matrix.row, dtype="<i4").tofile(handle)
        np.asarray(p_matrix.col, dtype="<i4").tofile(handle)
        np.asarray(p_matrix.data, dtype="<f4").tofile(handle)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # Keep the R benchmark alive and preserve the reason.
        print(f"openTSNE affinity validation failed: {exc}", file=sys.stderr)
        raise
