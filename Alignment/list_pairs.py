#!/usr/bin/env python3
"""list_pairs.py

Create a pairing manifest between Network_Layer PCAP/PCAPNG files and Lower_Layer telemetry TXT
files in the NetsLab-5GORAN-IDD archive.

This version is a refinement of earlier list_pairs iterations.

I fixed this
-------------------
previously, the default "conditional" tcp_ aliasing would *duplicate* a telemetry TXT into both:
  - its original key: tcp_X
  - its aliased key: X

That prevented wrong double-pairing, but it could still produce an extra "radio_only" group for
"tcp_X" even when the file had already been successfully paired via "X".

here,  *conditional* tcp_ aliasing is implemented as a **rewrite**:
  - If alias condition is met, the TXT is assigned ONLY to key X (not also tcp_X)

This makes the summary accounting cleaner and avoids confusing "radio_only" groups created only
by alias bookkeeping.

Key behavior
------------
1) Searches recursively under Network_Layer/Lower_Layer (Path.rglob)
2) Never chooses an arbitrary "first match" when multiple candidates exist.
   - if both sides exist but are not 1-to-1 => status becomes "ambiguous"
3) tcp_ alias modes:
   - off:        no aliasing
   - conditional (default): rewrite tcp_X -> X only when:
          (a) there is NO network pcap for tcp_X, and
          (b) there IS a network pcap for X
   - always:     add BOTH keys (tcp_X and X). This can create ambiguity; use only for debugging.

Outputs (written to --out-dir)
------------------------------
- pairs_collapsed.csv
- summary.csv
- unmatched_report.txt
- pairs_expanded.csv (optional; --expand-ambiguous)

Usage
-----
python3 list_pairs.py --root /path/to/archive --out-dir ./pairing_out

"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


FAMILY_DEFAULT = ["Benign", "BruteForce", "DDOS", "DoS", "Probe", "Web"]


def canon(stem: str) -> str:
    """Normalize stems to maximize cross-folder matching without being too aggressive."""
    s = stem.lower()
    s = s.replace("-", "_")
    s = re.sub(r"__+", "_", s).strip("_")
    # observed typo normalization (kept from your original script)
    s = s.replace("npig", "nping")
    return s


def list_files(dirpath: Path | None, exts: Set[str], recursive: bool = True) -> List[Path]:
    """Return matching files under dirpath."""
    if dirpath is None or not dirpath.exists():
        return []

    out: List[Path] = []
    it: Iterable[Path] = dirpath.rglob("*") if recursive else dirpath.iterdir()

    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        suff = p.suffix.lower().lstrip(".")
        if suff in exts:
            out.append(p)

    return sorted(out, key=lambda x: str(x))


def find_dir(inner: Path, names: Set[str]) -> Path | None:
    """Find a subdirectory matching one of the candidate names.

    Prefer direct children; fall back to the shallowest match anywhere under `inner`.
    """
    try:
        for p in inner.iterdir():
            if p.is_dir() and p.name.lower() in names:
                return p
    except FileNotFoundError:
        return None

    candidates = [p for p in inner.rglob("*") if p.is_dir() and p.name.lower() in names]
    if not candidates:
        return None

    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def uniq_paths(paths: Sequence[Path]) -> List[Path]:
    """Deduplicate while keeping deterministic order."""
    seen: Set[str] = set()
    out: List[Path] = []
    for p in sorted(paths, key=lambda x: str(x)):
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


@dataclass
class StemGroup:
    family: str
    canon_stem: str
    pcaps: List[Path]
    txts: List[Path]


def make_groups(
    family: str,
    pcaps: Sequence[Path],
    txts: Sequence[Path],
    tcp_alias_mode: str = "conditional",
) -> Tuple[List[StemGroup], int]:
    """Build canonical stem groups.

    tcp_alias_mode:
      - off: no aliasing
      - conditional: rewrite tcp_X -> X only when it helps match (default)
      - always: also add tcp_X -> X in addition to tcp_X (can create duplicates/ambiguity)

    Returns:
      groups, alias_applied_count
    """
    if tcp_alias_mode not in {"off", "conditional", "always"}:
        raise ValueError("tcp_alias_mode must be one of: off, conditional, always")

    # Build pcap map first so we can do conditional alias decisions safely.
    pmap: Dict[str, List[Path]] = defaultdict(list)
    for p in pcaps:
        pmap[canon(p.stem)].append(p)

    tmap: Dict[str, List[Path]] = defaultdict(list)
    alias_applied = 0

    for t in txts:
        k = canon(t.stem)

        # Determine which key(s) this TXT contributes to.
        keys_for_t: List[str] = [k]

        if tcp_alias_mode != "off" and k.startswith("tcp_"):
            k2 = k[len("tcp_") :]

            if tcp_alias_mode == "always":
                # add both keys
                keys_for_t = [k, k2]
                alias_applied += 1

            else:
                # conditional: REWRITE to k2 only when it increases matching
                do_alias = (k not in pmap) and (k2 in pmap)
                if do_alias:
                    keys_for_t = [k2]  # rewrite: do NOT also keep k
                    alias_applied += 1

        for kk in keys_for_t:
            tmap[kk].append(t)

    keys = sorted(set(pmap.keys()) | set(tmap.keys()))

    groups: List[StemGroup] = []
    for k in keys:
        groups.append(
            StemGroup(
                family=family,
                canon_stem=k,
                pcaps=uniq_paths(pmap.get(k, [])),
                txts=uniq_paths(tmap.get(k, [])),
            )
        )

    return groups, alias_applied


def status_for(group: StemGroup) -> str:
    has_p = len(group.pcaps) > 0
    has_t = len(group.txts) > 0

    if has_p and has_t:
        if len(group.pcaps) == 1 and len(group.txts) == 1:
            return "paired"
        return "ambiguous"
    if has_p and not has_t:
        return "network_only"
    return "radio_only"


def collapse_row(group: StemGroup) -> dict:
    st = status_for(group)
    return {
        "family": group.family,
        "canon_stem": group.canon_stem,
        "status": st,
        "n_pcaps": len(group.pcaps),
        "n_txts": len(group.txts),
        "pcap_paths": "|".join(str(p) for p in group.pcaps),
        "txt_paths": "|".join(str(t) for t in group.txts),
        "note": (
            (f"multiple_pcaps={len(group.pcaps)};" if len(group.pcaps) > 1 else "")
            + (f"multiple_txts={len(group.txts)};" if len(group.txts) > 1 else "")
        ),
    }


def expand_rows(group: StemGroup) -> List[dict]:
    """Enumerate candidate rows (does NOT decide correct pairing)."""
    st = status_for(group)
    rows: List[dict] = []

    if st in {"paired", "ambiguous"}:
        for p in group.pcaps:
            for t in group.txts:
                rows.append(
                    {
                        "family": group.family,
                        "canon_stem": group.canon_stem,
                        "status": st,
                        "pcap": str(p),
                        "txt": str(t),
                        "n_pcaps_in_group": len(group.pcaps),
                        "n_txts_in_group": len(group.txts),
                    }
                )
        return rows

    if st == "network_only":
        for p in group.pcaps:
            rows.append(
                {
                    "family": group.family,
                    "canon_stem": group.canon_stem,
                    "status": st,
                    "pcap": str(p),
                    "txt": "",
                    "n_pcaps_in_group": len(group.pcaps),
                    "n_txts_in_group": 0,
                }
            )
        return rows

    for t in group.txts:
        rows.append(
            {
                "family": group.family,
                "canon_stem": group.canon_stem,
                "status": st,
                "pcap": "",
                "txt": str(t),
                "n_pcaps_in_group": 0,
                "n_txts_in_group": len(group.txts),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to archive folder")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--families",
        default=",".join(FAMILY_DEFAULT),
        help=f"Comma-separated families (default: {','.join(FAMILY_DEFAULT)})",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search (mostly for debugging).",
    )
    ap.add_argument(
        "--expand-ambiguous",
        action="store_true",
        help="Also write pairs_expanded.csv (one row per candidate pair / file).",
    )
    ap.add_argument(
        "--tcp-alias-mode",
        choices=["off", "conditional", "always"],
        default="conditional",
        help="How to treat TXT stems starting with tcp_. Default: conditional (rewrite, safe).",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recursive = not args.no_recursive
    families = [x.strip() for x in args.families.split(",") if x.strip()]

    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}

    all_groups: List[StemGroup] = []
    summary_rows: List[dict] = []

    for fam in families:
        fam_dir = children.get(fam.lower())
        if fam_dir is None:
            continue

        # some archives have an extra nested folder repeating the family name
        inner_candidates = [
            p for p in fam_dir.iterdir() if p.is_dir() and p.name.lower() == fam.lower()
        ]
        inner = inner_candidates[0] if inner_candidates else fam_dir

        net_dir = find_dir(inner, {"network_layer", "network_data", "network"})
        low_dir = find_dir(inner, {"lower_layer", "lower", "radio", "radio_layer"})

        pcaps = list_files(net_dir, {"pcap", "pcapng"}, recursive=recursive)
        txts = list_files(low_dir, {"txt"}, recursive=recursive)

        groups, alias_applied = make_groups(
            fam,
            pcaps,
            txts,
            tcp_alias_mode=args.tcp_alias_mode,
        )
        all_groups.extend(groups)

        st_counts = defaultdict(int)
        for g in groups:
            st_counts[status_for(g)] += 1

        summary_rows.append(
            {
                "family": fam,
                "pcaps_found": len(pcaps),
                "txts_found": len(txts),
                "paired_groups": st_counts["paired"],
                "ambiguous_groups": st_counts["ambiguous"],
                "network_only_groups": st_counts["network_only"],
                "radio_only_groups": st_counts["radio_only"],
                "tcp_alias_applied": alias_applied,
                "tcp_alias_mode": args.tcp_alias_mode,
            }
        )

    df_collapsed = pd.DataFrame([collapse_row(g) for g in all_groups]).sort_values(
        ["family", "status", "canon_stem"]
    )
    collapsed_csv = out_dir / "pairs_collapsed.csv"
    df_collapsed.to_csv(collapsed_csv, index=False)

    expanded_csv = None
    if args.expand_ambiguous:
        expanded: List[dict] = []
        for g in all_groups:
            expanded.extend(expand_rows(g))
        df_expanded = pd.DataFrame(expanded).sort_values(
            ["family", "status", "canon_stem", "pcap", "txt"]
        )
        expanded_csv = out_dir / "pairs_expanded.csv"
        df_expanded.to_csv(expanded_csv, index=False)

    df_summary = pd.DataFrame(summary_rows).sort_values(["family"])
    summary_csv = out_dir / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    report_txt = out_dir / "unmatched_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        for fam in sorted(set(df_collapsed["family"].tolist())):
            f.write(f"=== {fam} ===\n")
            sub = df_collapsed[df_collapsed["family"] == fam]
            f.write(
                "counts: "
                + ", ".join(
                    f"{k}={int((sub.status == k).sum())}"
                    for k in ["paired", "ambiguous", "network_only", "radio_only"]
                )
                + "\n\n"
            )

            amb = sub[sub.status == "ambiguous"]
            if len(amb) > 0:
                f.write("Ambiguous (needs manual/heuristic resolution):\n")
                for _, r in amb.iterrows():
                    f.write(f"  {r.canon_stem}\n")
                    f.write(f"    pcaps({r.n_pcaps}): {r.pcap_paths}\n")
                    f.write(f"    txts ({r.n_txts}): {r.txt_paths}\n")
                f.write("\n")

            net_only = sub[sub.status == "network_only"]
            if len(net_only) > 0:
                f.write("Network only:\n")
                for _, r in net_only.iterrows():
                    f.write(f"  {r.canon_stem}  pcaps({r.n_pcaps}): {r.pcap_paths}\n")
                f.write("\n")

            rad_only = sub[sub.status == "radio_only"]
            if len(rad_only) > 0:
                f.write("Radio only:\n")
                for _, r in rad_only.iterrows():
                    f.write(f"  {r.canon_stem}  txts({r.n_txts}): {r.txt_paths}\n")
                f.write("\n")

        f.write("\n")

    print(f"Wrote: {collapsed_csv}")
    if expanded_csv is not None:
        print(f"Wrote: {expanded_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {report_txt}")


if __name__ == "__main__":
    main()
