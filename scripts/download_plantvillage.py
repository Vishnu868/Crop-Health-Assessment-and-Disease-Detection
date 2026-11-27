#!/usr/bin/env python3
"""Download a balanced 5-class subset of the PlantVillage dataset.

The original project trained YOLOv5 on five classes (Bacterial_spot, Black_rot,
Healthy, Late_Blight, Powder_Mildew). Its test-image filenames (e.g.
``UMD_Powd.M``, ``FAM_B.Rot``, ``Rutg._Bact.S``, ``GHLB2 Leaf``, ``RS_HL``) are
PlantVillage filenames, so we rebuild the same 5 super-classes from the public
PlantVillage repository (https://github.com/spMohanty/PlantVillage-Dataset).

For every image we fetch both the colour photo (``raw/color``) and the
background-removed version (``raw/segmented``); the latter is used to derive a
bounding box for the YOLO detector and a leaf mask for the health index.

Robustness:
* resumable: existing, non-empty files are skipped;
* per-request timeout + retries with exponential back-off;
* a thread pool with bounded futures (a stuck request cannot hang the run);
* a manifest CSV is written so the exact subset is reproducible.

Usage:
    python scripts/download_plantvillage.py --out data/raw --per-class 1600
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"

# super-class -> PlantVillage source folders
CLASS_MAP: dict[str, list[str]] = {
    "Bacterial_spot": ["Pepper,_bell___Bacterial_spot", "Peach___Bacterial_spot", "Tomato___Bacterial_spot"],
    "Black_rot": ["Apple___Black_rot", "Grape___Black_rot"],
    "Healthy": ["Apple___healthy", "Cherry_(including_sour)___healthy", "Grape___healthy", "Peach___healthy",
                "Pepper,_bell___healthy", "Potato___healthy", "Tomato___healthy"],
    "Late_Blight": ["Potato___Late_blight", "Tomato___Late_blight"],
    "Powder_Mildew": ["Cherry_(including_sour)___Powdery_mildew", "Squash___Powdery_mildew"],
}
CLASSES = list(CLASS_MAP)


def list_tree(clone_dir: Path) -> dict[str, list[str]]:
    """Return {source_folder: [file names]} using a blob-less shallow clone (metadata only, ~MBs)."""
    if not (clone_dir / ".git").exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO, str(clone_dir)],
                       check=True)
    out = {}
    for sources in CLASS_MAP.values():
        for src in sources:
            res = subprocess.run(["git", "-C", str(clone_dir), "ls-tree", "--name-only", "HEAD", f"raw/color/{src}/"],
                                 check=True, capture_output=True, text=True)
            out[src] = sorted(os.path.basename(p) for p in res.stdout.splitlines() if p.strip())
    return out


def balanced_sample(tree: dict[str, list[str]], per_class: int, seed: int) -> list[tuple[str, str, str]]:
    """Pick ``per_class`` images per super-class, spread as evenly as possible over its source crops."""
    rng = random.Random(seed)
    rows = []
    for cls, sources in CLASS_MAP.items():
        pools = {s: rng.sample(tree[s], len(tree[s])) for s in sources}
        quota = {s: 0 for s in sources}
        remaining = per_class
        # water-filling: give each source an equal share, redistribute what small sources can't fill
        active = [s for s in sources if pools[s]]
        while remaining > 0 and active:
            share = max(1, remaining // len(active))
            nxt = []
            for s in active:
                take = min(share, len(pools[s]) - quota[s], remaining)
                quota[s] += take
                remaining -= take
                if quota[s] < len(pools[s]):
                    nxt.append(s)
                if remaining == 0:
                    break
            active = nxt
        for s in sources:
            rows += [(cls, s, f) for f in pools[s][:quota[s]]]
    return rows


def fetch(url: str, dest: Path, retries: int = 5, timeout: float = 30.0) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dest)
            return "ok"
        except Exception as e:  # noqa: BLE001 - network errors of all kinds are retried
            err = e
            time.sleep(min(30, 2 ** attempt))
    if tmp.exists():
        tmp.unlink()
    return f"fail: {err}"


def safe_name(src: str, fname: str) -> str:
    stem = os.path.splitext(fname)[0].replace(" ", "_")
    crop = src.split("___")[0].replace(",", "").replace("(", "").replace(")", "")
    return f"{crop}__{stem}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--per-class", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-segmented", action="store_true", help="skip the background-removed images")
    args = ap.parse_args()

    out = Path(args.out)
    tree = list_tree(out / "_pv_index")
    rows = balanced_sample(tree, args.per_class, args.seed)

    jobs = []
    manifest = []
    for cls, src, fname in rows:
        name = safe_name(src, fname)
        q = urllib.parse.quote
        color_url = RAW + f"raw/color/{q(src)}/{q(fname)}"
        jobs.append((color_url, out / "color" / cls / name))
        seg_name = os.path.splitext(fname)[0] + "_final_masked.jpg"
        if not args.no_segmented:
            jobs.append((RAW + f"raw/segmented/{q(src)}/{q(seg_name)}", out / "segmented" / cls / name))
        manifest.append({"class": cls, "source": src, "original_file": fname, "file": name})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "source", "original_file", "file"])
        w.writeheader()
        w.writerows(manifest)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, u, d): (u, d) for u, d in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result(timeout=300)
            except Exception as e:  # noqa: BLE001
                r = f"fail: {e}"
            key = r.split(":")[0]
            stats[key] += 1
            if key == "fail":
                failures.append((futs[fu][0], r))
            if i % 500 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {stats} {time.time() - t0:.0f}s", flush=True)
    if failures:
        with open(out / "failed.txt", "w") as f:
            f.writelines(f"{u}\t{r}\n" for u, r in failures)
        print(f"{len(failures)} downloads failed (see failed.txt); re-run to resume.", file=sys.stderr)
    print("done", stats)


if __name__ == "__main__":
    main()
