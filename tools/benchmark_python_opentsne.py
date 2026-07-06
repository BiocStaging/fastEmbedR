#!/usr/bin/env python3
"""Benchmark the Python openTSNE package on local fastEmbedR datasets.

The script converts the RData dataset to a simple NumPy-readable binary cache
once, then measures total openTSNE wall-clock time. It is intentionally a
total-call benchmark: nearest-neighbour search, affinity construction,
initialization, and optimization are all included because Python openTSNE's
public TSNE.fit() call owns those stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--data-root", default="/Users/stefano/Documents/fastEmbedR/Data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--cache-dir", default="results/python_opentsne_cache")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--early-exaggeration-iter", type=int, default=250)
    p.add_argument("--negative-gradient-method", default="fft")
    p.add_argument("--neighbors", default="auto")
    p.add_argument("--initialization", default="pca")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_cache(data_root: Path, dataset: str, cache_dir: Path) -> tuple[Path, Path, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = cache_dir / f"{dataset}_float32_rowmajor.bin"
    y_path = cache_dir / f"{dataset}_labels_int32.bin"
    meta_path = cache_dir / f"{dataset}_meta.json"
    if x_path.exists() and y_path.exists() and meta_path.exists():
        return x_path, y_path, json.loads(meta_path.read_text())

    rdata = data_root / dataset / f"{dataset}.RData"
    if not rdata.exists():
        hits = sorted((data_root / dataset).glob("*.RData"))
        if not hits:
            raise FileNotFoundError(f"No RData file found under {data_root / dataset}")
        rdata = hits[0]

    r_code = f"""
    load({str(rdata)!r})
    if (!exists("dataset")) {{
      objs <- mget(ls())
      dataset <- NULL
      for (obj in objs) {{
        if (is.list(obj) && !is.null(obj$data)) {{ dataset <- obj; break }}
      }}
      if (is.null(dataset)) stop("No dataset list with $data found")
    }}
    x <- as.matrix(dataset$data)
    storage.mode(x) <- "double"
    labels <- dataset$labels
    if (is.null(labels)) labels <- rep(NA_integer_, nrow(x))
    lab <- as.integer(as.factor(labels))
    con <- file({str(x_path)!r}, "wb")
    writeBin(as.numeric(t(x)), con, size = 4, endian = "little")
    close(con)
    con <- file({str(y_path)!r}, "wb")
    writeBin(as.integer(lab), con, size = 4, endian = "little")
    close(con)
    meta <- list(n = nrow(x), p = ncol(x), rdata = {str(rdata)!r})
    if (!requireNamespace("jsonlite", quietly = TRUE)) {{
      writeLines(paste0('{{"n":', nrow(x), ',"p":', ncol(x), ',"rdata":"', {str(rdata)!r}, '"}}'), {str(meta_path)!r})
    }} else {{
      jsonlite::write_json(meta, {str(meta_path)!r}, auto_unbox = TRUE, pretty = TRUE)
    }}
    """
    run(["Rscript", "-e", r_code])
    return x_path, y_path, json.loads(meta_path.read_text())


def load_cached_matrix(x_path: Path, y_path: Path, meta: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    full_n = int(meta["n"])
    p = int(meta["p"])
    n = full_n if n <= 0 else min(n, full_n)
    x = np.memmap(x_path, dtype="<f4", mode="r", shape=(full_n, p))
    labels = np.memmap(y_path, dtype="<i4", mode="r", shape=(full_n,))
    # Force a contiguous in-memory slice for openTSNE. This time is counted as
    # load/setup, not as openTSNE fit time.
    return np.asarray(x[:n], dtype=np.float32, order="C"), np.asarray(labels[:n], dtype=np.int32)


def save_plot(embedding: np.ndarray, labels: np.ndarray, path: Path) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping plot", flush=True)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 7), dpi=160)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=1.5, cmap="tab10", linewidths=0)
    plt.xticks([])
    plt.yticks([])
    plt.title("Python openTSNE")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir or f"results/python_opentsne_{args.dataset}_{args.n}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    x_path, y_path, meta = ensure_cache(Path(args.data_root), args.dataset, cache_dir)
    load_t0 = time.perf_counter()
    x, labels = load_cached_matrix(x_path, y_path, meta, args.n)
    load_sec = time.perf_counter() - load_t0

    import openTSNE

    tsne = openTSNE.TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        early_exaggeration_iter=args.early_exaggeration_iter,
        early_exaggeration="auto",
        n_iter=args.n_iter,
        exaggeration=None,
        initialization=args.initialization,
        metric="euclidean",
        n_jobs=args.threads,
        neighbors=args.neighbors,
        negative_gradient_method=args.negative_gradient_method,
        random_state=args.seed,
        verbose=True,
    )

    fit_t0 = time.perf_counter()
    embedding = np.asarray(tsne.fit(x))
    fit_sec = time.perf_counter() - fit_t0
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    maxrss_gb = maxrss / (1024**3) if sys.platform == "darwin" else maxrss / (1024**2)

    np.save(out_dir / "embedding.npy", embedding)
    save_plot(embedding, labels, out_dir / "python_opentsne_embedding.png")

    row = {
        "dataset": args.dataset,
        "n": x.shape[0],
        "p": x.shape[1],
        "package": "openTSNE",
        "openTSNE_version": getattr(openTSNE, "__version__", "NA"),
        "perplexity": args.perplexity,
        "threads": args.threads,
        "seed": args.seed,
        "neighbors": args.neighbors,
        "negative_gradient_method": args.negative_gradient_method,
        "initialization": args.initialization,
        "early_exaggeration_iter": args.early_exaggeration_iter,
        "n_iter": args.n_iter,
        "load_sec": round(load_sec, 6),
        "fit_sec": round(fit_sec, 6),
        "total_sec": round(load_sec + fit_sec, 6),
        "maxrss_gb": round(maxrss_gb, 6),
        "plot_file": str(out_dir / "python_opentsne_embedding.png"),
    }
    with open(out_dir / "python_opentsne_benchmark.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
