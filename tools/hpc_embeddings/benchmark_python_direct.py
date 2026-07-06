#!/usr/bin/env python3
"""Run Python reference embeddings in a native Python process.

The R publication benchmark uses this helper to separate package runtime from
the R/reticulate boundary. The input is a pre-exported NPZ file generated before
the timed section of the R worker. This script records Python-side fit time with
``time.perf_counter()``, saves the 2D layout as CSV, and writes a small JSON
metadata file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=[
                            "python_opentsne_fft_direct",
                            "python_umap_learn_direct",
                            "rapids_cuml_umap_full_direct",
                            "rapids_cuml_tsne_full_direct",
                        ])
    parser.add_argument("--input", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--perplexity", type=float, default=15.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def to_host_array(x):
    """Convert NumPy/CuPy/cuDF-like outputs to a NumPy array."""
    if hasattr(x, "to_numpy"):
        try:
            x = x.to_numpy()
        except Exception:
            pass
    try:
        import cupy as cp  # type: ignore
        try:
            x = cp.asnumpy(x)
        except Exception:
            pass
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def run_opentsne(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import openTSNE

    model = openTSNE.TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        initialization="pca",
        negative_gradient_method="fft",
        n_iter=500,
        early_exaggeration_iter=250,
        random_state=int(args.seed),
        n_jobs=int(args.threads),
        verbose=False,
    )
    return to_host_array(model.fit(x))


def run_umap_learn(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap

    model = umap.UMAP(
        n_neighbors=int(args.k),
        n_components=2,
        metric="euclidean",
        random_state=int(args.seed),
        n_jobs=int(args.threads),
        verbose=False,
    )
    return to_host_array(model.fit_transform(x))


def run_cuml_umap(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from cuml import UMAP

    model = UMAP(
        n_neighbors=int(args.k),
        n_components=2,
        metric="euclidean",
        random_state=int(args.seed),
        verbose=False,
    )
    return to_host_array(model.fit_transform(x))


def run_cuml_tsne(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from cuml.manifold import TSNE

    base = dict(
        n_components=2,
        perplexity=float(args.perplexity),
        random_state=int(args.seed),
        verbose=False,
    )
    try:
        model = TSNE(**base, method="fft")
    except TypeError:
        model = TSNE(**base)
    return to_host_array(model.fit_transform(x))


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(args.threads))

    meta = {
        "method": args.method,
        "status": "failed",
        "python_fit_sec": None,
        "error": None,
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.layout).parent.mkdir(parents=True, exist_ok=True)

    try:
        with np.load(args.input) as data:
            x = np.asarray(data["x"], dtype=np.float32, order="C")

        runners = {
            "python_opentsne_fft_direct": run_opentsne,
            "python_umap_learn_direct": run_umap_learn,
            "rapids_cuml_umap_full_direct": run_cuml_umap,
            "rapids_cuml_tsne_full_direct": run_cuml_tsne,
        }
        start = time.perf_counter()
        layout = runners[args.method](x, args)
        fit_sec = time.perf_counter() - start

        layout = np.asarray(layout, dtype=np.float32)
        if layout.ndim != 2 or layout.shape[1] < 2:
            raise RuntimeError(f"Expected a 2D embedding with >=2 columns, got shape {layout.shape}")
        layout = layout[:, :2]
        np.savetxt(args.layout, layout, delimiter=",")
        meta.update(
            status="success",
            python_fit_sec=float(fit_sec),
            n=int(x.shape[0]),
            p=int(x.shape[1]),
            layout_file=str(args.layout),
            python_executable=sys.executable,
        )
    except Exception as exc:  # pragma: no cover - failure metadata is the API.
        meta["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return 1

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
