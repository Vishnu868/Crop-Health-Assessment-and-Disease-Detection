#!/usr/bin/env python3
"""Raw PlantVillage subset -> splits, classifier arrays, YOLO dataset, health indices.

Outputs
  data/processed/splits.csv        one row per image: class, split, YOLO box, health indices
  data/processed/pv128.npz         uint8 arrays X (colour), Xseg (background removed), y, split
  data/yolo/{images,labels}/{train,val,test}  YOLO-format detection dataset (+ data.yaml)

Every image is processed independently in a worker process; results are
collected with a timeout so one bad file cannot hang the pool.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, IMG_SIZE, PROCESSED, RAW, SEED, YOLO_DIR  # noqa: E402
from src.health import health_indices, leaf_mask, mask_to_box  # noqa: E402


def load_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process(args: tuple[str, str, str]) -> dict | None:
    raw, cls, fname = args
    rgb = load_rgb(Path(raw) / "color" / cls / fname)
    if rgb is None:
        return None
    seg_path = Path(raw) / "segmented" / cls / fname
    seg = load_rgb(seg_path) if seg_path.exists() else None
    if seg is not None and seg.shape != rgb.shape:
        seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]))
    mask = leaf_mask(rgb, seg)
    box = mask_to_box(mask)
    if box is None or box[2] * box[3] < 0.02:  # fall back to the full frame for empty/garbage masks
        box = (0.5, 0.5, 1.0, 1.0)
    hi = health_indices(rgb, mask)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    seg_rgb = rgb.copy()
    seg_rgb[~mask] = 0
    seg_small = cv2.resize(seg_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return {"class": cls, "file": fname, "has_segmented": seg is not None,
            "xc": box[0], "yc": box[1], "bw": box[2], "bh": box[3], **hi,
            "_x": small, "_xs": seg_small}


def _health_job(args: tuple[str, str, str]) -> dict:
    raw, cls, fname = args
    rgb = load_rgb(Path(raw) / "color" / cls / fname)
    seg_path = Path(raw) / "segmented" / cls / fname
    seg = load_rgb(seg_path) if seg_path.exists() else None
    if seg is not None and seg.shape != rgb.shape:
        seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]))
    return {"file": fname, **health_indices(rgb, leaf_mask(rgb, seg))}


def update_health(raw: Path, workers: int) -> None:
    df = pd.read_csv(PROCESSED / "splits.csv")
    jobs = [(str(raw), r["class"], r["file"]) for _, r in df.iterrows()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [f.result(timeout=120) for f in [ex.submit(_health_job, j) for j in jobs]]
    h = pd.DataFrame(rows).set_index("file")
    for col in h.columns:
        df[col] = df["file"].map(h[col])
    df.to_csv(PROCESSED / "splits.csv", index=False)
    print(df.groupby("class")[["health_score", "vari", "exg"]].median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--health-only", action="store_true",
                    help="only recompute the health-index columns of an existing splits.csv")
    args = ap.parse_args()
    raw = Path(args.raw)
    if args.health_only:
        update_health(raw, args.workers)
        return

    man = pd.read_csv(raw / "manifest.csv")
    jobs = [(str(raw), r["class"], r["file"]) for _, r in man.iterrows()]
    rows, X, XS = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for i, fu in enumerate(futs):
            try:
                r = fu.result(timeout=120)
            except Exception as e:  # noqa: BLE001
                print("skip", jobs[i][2], e)
                r = None
            if r is None:
                continue
            X.append(r.pop("_x"))
            XS.append(r.pop("_xs"))
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"processed {i + 1}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    df = df.merge(man[["file", "source"]], on="file", how="left")
    df["label"] = df["class"].map({c: i for i, c in enumerate(CLASSES)})
    # stratify on class x source crop so every split has the same crop mix
    strat = df["class"] + "|" + df["source"]
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=strat, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=strat.iloc[tmp], random_state=SEED)
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    df["split"] = split

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "splits.csv", index=False)
    np.savez_compressed(PROCESSED / "pv128.npz", X=np.stack(X), Xseg=np.stack(XS),
                        y=df["label"].to_numpy(np.int64), split=df["split"].to_numpy(str))

    # YOLO detection dataset (symlinks to the colour images + one box per image)
    for _, r in df.iterrows():
        img_dir = YOLO_DIR / "images" / r["split"]
        lab_dir = YOLO_DIR / "labels" / r["split"]
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        dst = img_dir / r["file"]
        if not dst.exists():
            os.symlink((raw / "color" / r["class"] / r["file"]).resolve(), dst)
        (lab_dir / (Path(r["file"]).stem + ".txt")).write_text(
            f"{r['label']} {r['xc']:.6f} {r['yc']:.6f} {r['bw']:.6f} {r['bh']:.6f}\n")
    (YOLO_DIR / "data.yaml").write_text(
        f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    print(df.groupby(["split", "class"]).size().unstack())
    print("images without segmented mask:", int((~df["has_segmented"]).sum()))


if __name__ == "__main__":
    main()
