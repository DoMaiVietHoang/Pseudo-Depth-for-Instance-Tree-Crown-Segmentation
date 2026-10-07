"""Aggregate metrics across multiple seeds of the same config.

Reads `work_dirs/<config>__seed<N>/` directories produced by run_ablations.sh
and emits a markdown table with mean ± std for each config that has 2+ seeds.
Single-seed configs are shown with the raw value.

Usage:
    python tools/eval/aggregate_seeds.py work_dirs --out results_meanstd.md
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import re
from collections import defaultdict
from glob import glob


METRIC_KEYS = [
    "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
    "coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="parent dir containing <config>__seed<N>/")
    p.add_argument("--out", default="results_meanstd.md")
    return p.parse_args()


def parse_scalars(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return rows


def best_metrics(rows):
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = max(vals)
    return out


def find_boundary_ap(d):
    for p in glob(osp.join(d, "**", "boundary_ap.txt"), recursive=True):
        try:
            txt = open(p).read()
        except OSError:
            continue
        m = re.search(r"Boundary-AP \(mean[^=]*=\s*([0-9.]+)", txt)
        if m:
            return float(m.group(1))
    return None


def scan_dir(d):
    cands = sorted(glob(osp.join(d, "*", "vis_data", "scalars.json")))
    if not cands:
        cands = sorted(glob(osp.join(d, "vis_data", "scalars.json")))
    rows = []
    for c in cands:
        rows.extend(parse_scalars(c))
    if not rows:
        return None
    out = best_metrics(rows)
    b = find_boundary_ap(d)
    if b is not None:
        out["boundary_AP"] = b
    return out


def split_seed(name):
    m = re.match(r"(.+?)__seed(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return name, None


def fmt_meanstd(vals, scale=100, prec=2):
    import statistics
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0] * scale:.{prec}f}"
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    return f"{mu * scale:.{prec}f} ± {sd * scale:.{prec}f}"


def main():
    args = parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(glob(osp.join(args.root, "*"))):
        if not osp.isdir(d):
            continue
        name, seed = split_seed(osp.basename(d))
        m = scan_dir(d)
        if m is None:
            continue
        groups[name].append(m)

    if not groups:
        print("no runs found")
        return

    metric_cols = [
        ("bbox AP", "coco/bbox_mAP"),
        ("bbox AP50", "coco/bbox_mAP_50"),
        ("segm AP", "coco/segm_mAP"),
        ("segm AP50", "coco/segm_mAP_50"),
        ("segm AP75", "coco/segm_mAP_75"),
        ("Bnd-AP", "boundary_AP"),
    ]

    headers = ["Config", "n_seeds"] + [c[0] for c in metric_cols]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for name, runs in sorted(groups.items()):
        cells = [name, str(len(runs))]
        for _, key in metric_cols:
            vals = [r[key] for r in runs if key in r]
            cells.append(fmt_meanstd(vals))
        lines.append("| " + " | ".join(cells) + " |")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(groups)} configs)")
    for name, runs in sorted(groups.items()):
        print(f"  {name}: {len(runs)} seed(s)")


if __name__ == "__main__":
    main()
