"""Gather metric numbers from all work_dirs and emit a CSV + Markdown table.

Reads mmengine's `scalars.json` (or fallback to scanning .log files) from each
work_dir, picks the best val-time COCO bbox/segm mAP, and optionally joins
Boundary-AP results from boundary_ap.py output.

Usage:
    python tools/eval/aggregate_results.py work_dirs --out results.md
    python tools/eval/aggregate_results.py work_dirs/*ablations* --out abl.md --csv abl.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import re
from glob import glob


METRIC_KEYS = [
    "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
    "coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75",
]
GATE_KEYS = ["gate/L1", "gate/L2", "gate/L3", "gate/mean"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="work_dirs to scan (glob OK)")
    p.add_argument("--out", default="results.md", help="markdown output path")
    p.add_argument("--csv", help="optional CSV output path")
    return p.parse_args()


def expand(dirs):
    out = []
    for d in dirs:
        if any(c in d for c in "*?["):
            out.extend(glob(d))
        else:
            out.append(d)
    return [d for d in sorted(set(out)) if osp.isdir(d)]


def parse_scalars_json(path):
    """Return list of dicts (one per logged step)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def best_metrics(rows):
    """Return dict of best (max) values seen for each METRIC_KEYS, and the last
    seen value for each GATE_KEYS."""
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = max(vals)
    for k in GATE_KEYS:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = vals[-1]
    # Track best epoch for segm_mAP
    best_step = -1
    best_val = -1.0
    for r in rows:
        v = r.get("coco/segm_mAP")
        if v is not None and v > best_val:
            best_val = v
            best_step = r.get("step", -1) or r.get("epoch", -1)
    if best_step >= 0:
        out["best_step"] = best_step
    return out


def scan_workdir(d):
    # mmengine writes scalars to <work_dir>/<timestamp>/vis_data/scalars.json
    cands = sorted(glob(osp.join(d, "*", "vis_data", "scalars.json")))
    if not cands:
        # fallback: nested only one level
        cands = sorted(glob(osp.join(d, "vis_data", "scalars.json")))
    if not cands:
        return None
    rows = []
    for c in cands:
        rows.extend(parse_scalars_json(c))
    if not rows:
        return None
    return best_metrics(rows)


def find_boundary_ap(d):
    """Look for a boundary_ap.txt or boundary_ap.json next to results.pkl."""
    for name in ("boundary_ap.json", "boundary_ap.txt"):
        for p in glob(osp.join(d, "**", name), recursive=True):
            try:
                if name.endswith(".json"):
                    return json.loads(open(p).read())
                txt = open(p).read()
                m = re.search(
                    r"Boundary-AP \(mean[^=]*=\s*([0-9.]+)", txt)
                if m:
                    out = {"boundary_AP": float(m.group(1))}
                    m50 = re.search(r"Boundary-AP @ 0\.50\s*=\s*([0-9.]+)", txt)
                    m75 = re.search(r"Boundary-AP @ 0\.75\s*=\s*([0-9.]+)", txt)
                    if m50: out["boundary_AP_50"] = float(m50.group(1))
                    if m75: out["boundary_AP_75"] = float(m75.group(1))
                    return out
            except Exception:
                continue
    return {}


def fmt_pct(v):
    if v is None:
        return "-"
    return f"{v * 100:.2f}"


def fmt_f(v):
    if v is None:
        return "-"
    return f"{v:.3f}"


def main():
    args = parse_args()
    dirs = expand(args.dirs)
    if not dirs:
        print("no dirs matched")
        return

    rows = []
    for d in dirs:
        m = scan_workdir(d) or {}
        m.update(find_boundary_ap(d))
        rows.append({
            "name": osp.basename(d.rstrip("/")),
            "bbox_mAP": m.get("coco/bbox_mAP"),
            "bbox_AP50": m.get("coco/bbox_mAP_50"),
            "bbox_AP75": m.get("coco/bbox_mAP_75"),
            "segm_mAP": m.get("coco/segm_mAP"),
            "segm_AP50": m.get("coco/segm_mAP_50"),
            "segm_AP75": m.get("coco/segm_mAP_75"),
            "boundary_AP": m.get("boundary_AP"),
            "gate_L1": m.get("gate/L1"),
            "gate_L2": m.get("gate/L2"),
            "gate_L3": m.get("gate/L3"),
            "gate_mean": m.get("gate/mean"),
            "best_epoch": m.get("best_step"),
        })

    # Markdown table
    headers = ["Config", "bbox AP", "AP50", "AP75",
               "segm AP", "AP50", "AP75", "Bnd-AP",
               "g L1", "g L2", "g L3", "best ep"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["name"],
            fmt_pct(r["bbox_mAP"]), fmt_pct(r["bbox_AP50"]), fmt_pct(r["bbox_AP75"]),
            fmt_pct(r["segm_mAP"]), fmt_pct(r["segm_AP50"]), fmt_pct(r["segm_AP75"]),
            fmt_pct(r["boundary_AP"]) if r["boundary_AP"] is not None else "-",
            fmt_f(r["gate_L1"]), fmt_f(r["gate_L2"]), fmt_f(r["gate_L3"]),
            str(r["best_epoch"]) if r["best_epoch"] is not None else "-",
        ]) + " |")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
