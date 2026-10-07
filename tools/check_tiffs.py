"""Scan a directory tree for corrupted / unreadable TIFF (and other image) files.

Usage:
    # Scan one folder
    python tools/check_tiffs.py /path/to/images

    # Scan multiple folders, save bad list to file, use 8 worker processes
    python tools/check_tiffs.py /data/BAMFOREST/images/train /data/BAMFOREST/images/val \
        --out bad_files.txt --workers 8

    # Include extra extensions and also try a full decode (slower, more thorough)
    python tools/check_tiffs.py /path/to/images --ext .tif .tiff .png --full-decode
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


DEFAULT_EXTS = (".tif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser(description="Find corrupted image files")
    p.add_argument("paths", nargs="+", help="files or directories to scan")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help="file extensions to include (case-insensitive)")
    p.add_argument("--out", help="write list of bad file paths to this text file")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="parallel worker processes (default: half of CPU cores)")
    p.add_argument("--full-decode", action="store_true",
                   help="decode the entire pixel buffer (slower; catches partial corruption)")
    p.add_argument("--delete", action="store_true",
                   help="DELETE bad files (use with care; prompts for confirmation)")
    p.add_argument("--quiet", action="store_true",
                   help="only print bad files, not progress")
    return p.parse_args()


def collect_files(paths: list[str], exts: tuple[str, ...]) -> list[str]:
    exts_lower = tuple(e.lower() for e in exts)
    files: list[str] = []
    for p in paths:
        if osp.isfile(p):
            if p.lower().endswith(exts_lower):
                files.append(p)
        elif osp.isdir(p):
            for root, _dirs, fnames in os.walk(p):
                for fn in fnames:
                    if fn.lower().endswith(exts_lower):
                        files.append(osp.join(root, fn))
        else:
            print(f"[check] warning: {p} not found, skipping")
    return sorted(set(files))


def check_one(path: str, full_decode: bool) -> tuple[str, str | None]:
    """Return (path, error_string) where error_string is None if file is OK."""
    # Always do the cheap checks first: existence + nonzero size.
    try:
        st = os.stat(path)
    except OSError as e:
        return path, f"stat failed: {e}"
    if st.st_size == 0:
        return path, "zero-byte file"

    # Try cv2 first (matches what the training pipeline uses via mmcv).
    try:
        import cv2
        import numpy as np
        # IMREAD_UNCHANGED keeps multi-channel / float; IMREAD_COLOR matches mmcv default
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return path, "cv2.imread returned None"
        if full_decode:
            # Touch every pixel to force the lazy decoder to actually read all strips.
            _ = np.asarray(img).sum()
    except Exception as e:
        return path, f"cv2 error: {type(e).__name__}: {e}"

    # PIL/Pillow is stricter about partial corruption — use as a second pass.
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()  # checks header / structure without decoding
        if full_decode:
            with Image.open(path) as im:
                im.load()  # forces full decode
    except Exception as e:
        return path, f"PIL error: {type(e).__name__}: {e}"

    return path, None


def main():
    args = parse_args()

    files = collect_files(args.paths, tuple(args.ext))
    if not files:
        print("[check] no files found")
        return 0

    if not args.quiet:
        print(f"[check] scanning {len(files)} files with {args.workers} workers "
              f"(full_decode={args.full_decode})")

    bad: list[tuple[str, str]] = []
    t0 = time.time()
    done = 0
    next_report = time.time() + 5.0

    if args.workers <= 1:
        for path in files:
            p, err = check_one(path, args.full_decode)
            done += 1
            if err:
                bad.append((p, err))
                print(f"BAD  {p}  -- {err}")
            if not args.quiet and time.time() >= next_report:
                print(f"[check] progress {done}/{len(files)} "
                      f"({done / len(files) * 100:.1f}%), bad so far: {len(bad)}")
                next_report = time.time() + 5.0
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(check_one, p, args.full_decode): p for p in files}
            for fut in as_completed(futures):
                p, err = fut.result()
                done += 1
                if err:
                    bad.append((p, err))
                    print(f"BAD  {p}  -- {err}")
                if not args.quiet and time.time() >= next_report:
                    print(f"[check] progress {done}/{len(files)} "
                          f"({done / len(files) * 100:.1f}%), bad so far: {len(bad)}")
                    next_report = time.time() + 5.0

    dt = time.time() - t0
    print()
    print(f"[check] done in {dt:.1f}s — {len(bad)} bad / {len(files)} total "
          f"({len(bad) / len(files) * 100:.2f}%)")

    if args.out and bad:
        with open(args.out, "w") as f:
            for p, err in sorted(bad):
                f.write(f"{p}\t{err}\n")
        print(f"[check] wrote bad list to {args.out}")

    if args.delete and bad:
        print()
        ans = input(f"DELETE {len(bad)} bad files? Type 'yes' to confirm: ").strip()
        if ans == "yes":
            for p, _ in bad:
                try:
                    os.remove(p)
                    print(f"deleted {p}")
                except OSError as e:
                    print(f"failed to delete {p}: {e}")
        else:
            print("aborted, nothing deleted")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
