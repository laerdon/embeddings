#!/usr/bin/env python3
"""Plot how cosine similarity and euclidean distance vary with words changed.

It expects CSV files produced by embedding_collision_test.py whose filenames follow
collision_results_<k>words_<n>var_<timestamp>.csv

Usage:
    python plot_similarity_trend.py --input-dir full_run --output trend.png
"""

import argparse
import glob
import os
import re
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

FILE_RE = re.compile(r"collision_results_(\d+)words_\d+var_.*\.csv")


def gather_files(directory: str) -> Dict[int, str]:
    """Return mapping words_changed -> filepath"""
    paths = glob.glob(os.path.join(directory, "collision_results_*words_*var_*.csv"))
    mapping = {}
    for p in paths:
        m = FILE_RE.search(os.path.basename(p))
        if m:
            k = int(m.group(1))
            mapping[k] = p
    return mapping


def compute_stats(path: str):
    df = pd.read_csv(path)
    mean_cos = df["cosine_similarity"].mean()
    mean_euc = df["euclidean_distance"].mean()
    return mean_cos, mean_euc


def main():
    parser = argparse.ArgumentParser(
        description="plot similarity trend vs words changed"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="collision_results",
        help="directory with result csvs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similarity_trend.png",
        help="output plot filename",
    )
    args = parser.parse_args()

    files = gather_files(args.input_dir)
    if not files:
        print(f"No result CSVs found in {args.input_dir}")
        return

    xs = sorted(files.keys())
    cos_vals = []
    euc_vals = []
    for k in xs:
        cos, euc = compute_stats(files[k])
        cos_vals.append(cos)
        euc_vals.append(euc)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("words changed per variant")
    ax1.set_ylabel("mean cosine similarity", color=color1)
    ax1.plot(xs, cos_vals, marker="o", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("mean euclidean distance", color=color2)
    ax2.plot(xs, euc_vals, marker="s", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Embedding sensitivity to word perturbations")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
