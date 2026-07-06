#!/usr/bin/env python3
"""Benchmark Python UMAP implementations on local fastEmbedR datasets.

Backends:
  * umap-learn: CPU Python implementation using pynndescent/numba.
  * cuml: RAPIDS cuML UMAP when available on a CUDA machine.

The script mirrors tools/benchmark_python_opentsne.py: it converts an RData
dataset into a row-major float32 cache once, then times the UMAP fit call.
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
    p.add_argument("--cache-dir", default="results/python_umap_cache")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--backend", choices=["umap-learn", "cuml", "both"], default="both")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", default="none", help="'none' allows multithreaded umap-learn; otherwise use an integer seed")
    p.add_argument("--n-epochs", type=int, default=0, help="0 means backend default")
    p.add_argument("--init", default="spectral")
    p.add_argument("--warmup", action="store_true", help="Run a small warmup fit before timing to avoid numba JIT in the measured run")
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
    return np.asarray(x[:n], dtype=np.float32, order="C"), np.asarray(labels[:n], dtype=np.int32)


def save_plot_r(embedding: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    csv_path = path.with_suffix(".embedding.csv")
    lab_path = path.with_suffix(".labels.csv")
    np.savetxt(csv_path, embedding, delimiter=",")
    np.savetxt(lab_path, labels, fmt="%d", delimiter=",")
    r_code = f"""
    emb <- as.matrix(read.csv({str(csv_path)!r}, header = FALSE))
    labels <- scan({str(lab_path)!r}, quiet = TRUE)
    png({str(path)!r}, width = 1600, height = 1300, res = 160)
    par(mar = c(4.2, 4.2, 3, 1))
    plot(emb, pch = 20, cex = 0.45, col = labels,
         xlab = "UMAP1", ylab = "UMAP2", main = {title!r})
    dev.off()
    """
    subprocess.run(["Rscript", "-e", r_code], check=False)


def maxrss_gb() -> float:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / (1024**3) if sys.platform == "darwin" else maxrss / (1024**2)


def seed_value(seed_text: str):
    if seed_text.lower() in {"none", "na", "null", "-1"}:
        return None
    return int(seed_text)


def benchmark_umap_learn(args: argparse.Namespace, x: np.ndarray, labels: np.ndarray, out_dir: Path) -> dict:
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    try:
        import umap
        try:
            import numba
            numba.set_num_threads(args.threads)
        except Exception:
            pass
    except Exception as exc:
        return failed_row(args, "umap-learn", "backend_unavailable", exc)

    random_state = seed_value(args.seed)
    kwargs = dict(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        init=args.init,
        n_jobs=args.threads,
        random_state=random_state,
        verbose=True,
    )
    if args.n_epochs > 0:
        kwargs["n_epochs"] = args.n_epochs

    if args.warmup:
        warm_n = min(2000, x.shape[0])
        print(f"Running umap-learn warmup on {warm_n} rows", flush=True)
        umap.UMAP(**kwargs).fit_transform(x[:warm_n])

    t0 = time.perf_counter()
    embedding = umap.UMAP(**kwargs).fit_transform(x)
    fit_sec = time.perf_counter() - t0

    np.save(out_dir / "umap_learn_embedding.npy", embedding)
    save_plot_r(embedding, labels, out_dir / "umap_learn_embedding.png", "Python umap-learn MNIST70k")
    return success_row(args, "umap-learn", getattr(umap, "__version__", "NA"), fit_sec, embedding, out_dir / "umap_learn_embedding.png")


def benchmark_cuml(args: argparse.Namespace, x: np.ndarray, labels: np.ndarray, out_dir: Path) -> dict:
    try:
        import cupy as cp
        import cuml
        from cuml import UMAP
    except Exception as exc:
        return failed_row(args, "cuml", "backend_unavailable", exc)

    random_state = seed_value(args.seed)
    kwargs = dict(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        init=args.init,
        random_state=random_state,
        verbose=True,
    )
    if args.n_epochs > 0:
        kwargs["n_epochs"] = args.n_epochs

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    x_gpu = cp.asarray(x, dtype=cp.float32)
    model = UMAP(**kwargs)
    embedding_gpu = model.fit_transform(x_gpu)
    cp.cuda.Stream.null.synchronize()
    fit_sec = time.perf_counter() - t0
    embedding = cp.asnumpy(embedding_gpu)

    np.save(out_dir / "cuml_umap_embedding.npy", embedding)
    save_plot_r(embedding, labels, out_dir / "cuml_umap_embedding.png", "RAPIDS cuML UMAP MNIST70k")
    return success_row(args, "cuml", getattr(cuml, "__version__", "NA"), fit_sec, embedding, out_dir / "cuml_umap_embedding.png")


def success_row(args: argparse.Namespace, backend: str, version: str, fit_sec: float, embedding: np.ndarray, plot_file: Path) -> dict:
    return {
        "dataset": args.dataset,
        "n": int(embedding.shape[0]),
        "p": "",
        "backend": backend,
        "status": "success",
        "version": version,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "threads": args.threads,
        "seed": args.seed,
        "init": args.init,
        "n_epochs": args.n_epochs if args.n_epochs > 0 else "default",
        "fit_sec": round(fit_sec, 6),
        "maxrss_gb": round(maxrss_gb(), 6),
        "plot_file": str(plot_file),
        "error": "",
    }


def failed_row(args: argparse.Namespace, backend: str, status: str, exc: Exception) -> dict:
    return {
        "dataset": args.dataset,
        "n": "",
        "p": "",
        "backend": backend,
        "status": status,
        "version": "",
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "threads": args.threads,
        "seed": args.seed,
        "init": args.init,
        "n_epochs": args.n_epochs if args.n_epochs > 0 else "default",
        "fit_sec": "",
        "maxrss_gb": round(maxrss_gb(), 6),
        "plot_file": "",
        "error": repr(exc),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir or f"results/python_umap_{args.dataset}_{args.n}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path, y_path, meta = ensure_cache(Path(args.data_root), args.dataset, Path(args.cache_dir))
    load_t0 = time.perf_counter()
    x, labels = load_cached_matrix(x_path, y_path, meta, args.n)
    load_sec = time.perf_counter() - load_t0

    rows: list[dict] = []
    if args.backend in {"umap-learn", "both"}:
        rows.append(benchmark_umap_learn(args, x, labels, out_dir))
    if args.backend in {"cuml", "both"}:
        rows.append(benchmark_cuml(args, x, labels, out_dir))

    for row in rows:
        row["n"] = row["n"] or x.shape[0]
        row["p"] = row["p"] or x.shape[1]
        row["load_sec"] = round(load_sec, 6)
        if row["fit_sec"] != "":
            row["total_sec"] = round(load_sec + float(row["fit_sec"]), 6)
        else:
            row["total_sec"] = ""

    csv_path = out_dir / "python_umap_benchmark.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
