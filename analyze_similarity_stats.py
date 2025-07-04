#!/usr/bin/env python3
"""Summarize cosine similarity statistics from collision result CSV files.

Usage:
    python analyze_similarity_stats.py --input-dir collision_results

It scans for files matching `collision_results_*words_*var_*.csv` and prints
mean, std, min, max of the `cosine_similarity` column for each file.
"""

import argparse
import glob
import os
import pandas as pd
from typing import List


def gather_csv_files(directory: str) -> List[str]:
    pattern = os.path.join(directory, "collision_results_*words_*var_*.csv")
    return sorted(glob.glob(pattern))


def summarize_file(path: str):
    df = pd.read_csv(path)
    if "cosine_similarity" not in df.columns:
        print(f"[skip] {os.path.basename(path)} has no cosine_similarity column")
        return

    stats = {
        "mean": df["cosine_similarity"].mean(),
        "std": df["cosine_similarity"].std(),
        "min": df["cosine_similarity"].min(),
        "max": df["cosine_similarity"].max(),
        "count": len(df),
    }
    # extract run info from filename if possible
    fname = os.path.basename(path)
    print(f"File: {fname}")
    print(
        f"  count={stats['count']}, mean={stats['mean']:.6f}, std={stats['std']:.6f},"
        f" min={stats['min']:.6f}, max={stats['max']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="summarize cosine similarity stats from collision result csvs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="collision_results",
        help="directory containing collision_results_*.csv files",
    )
    args = parser.parse_args()

    csv_files = gather_csv_files(args.input_dir)
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    for path in csv_files:
        summarize_file(path)


if __name__ == "__main__":
    main()
