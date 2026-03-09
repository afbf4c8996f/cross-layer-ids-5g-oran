#!/usr/bin/env python3
"""extract_paired_runs.py

Utility: read pairs_collapsed.csv produced by list_pairs_* scripts and export a clean
paired-runs manifest (one row per paired run):

  family, canon_stem, pcap_path, txt_path

It also writes a per-family count report to stdout.

Usage:
  python3 extract_paired_runs.py --pairs ./pairing_out/pairs_collapsed.csv --out ./paired_runs.csv

Optional:
  --family DDOS --family DoS   (repeatable) to filter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Path to pairs_collapsed.csv")
    ap.add_argument("--out", required=True, help="Output CSV path for paired runs")
    ap.add_argument(
        "--family",
        action="append",
        default=[],
        help="Family filter (repeatable). Example: --family DDOS --family DoS",
    )
    args = ap.parse_args()

    pairs_path = Path(args.pairs).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairs_path)

    if args.family:
        fam_set = {f.strip() for f in args.family if f.strip()}
        df = df[df["family"].isin(fam_set)].copy()

    df = df[df["status"] == "paired"].copy()

    # In paired rows, each of these should contain exactly one path.
    df["pcap_path"] = df["pcap_paths"].astype(str).str.split("|").str[0]
    df["txt_path"] = df["txt_paths"].astype(str).str.split("|").str[0]

    out = df[["family", "canon_stem", "pcap_path", "txt_path"]].sort_values(
        ["family", "canon_stem"]
    )

    out.to_csv(out_path, index=False)

    counts = out.groupby("family").size().sort_values(ascending=False)
    print("Paired runs written:", len(out))
    print(counts.to_string())
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
