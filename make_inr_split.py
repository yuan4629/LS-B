# -*- coding: utf-8 -*-
"""make_inr_split.py -- generate the frozen ImageNet-R train/test split used by the D2 experiments.

The repository expects the split at <data_root>/imagenet-r_split/{train.txt,test.txt}
(line format `<wnid>/<filename>`, see baselines/bilora_adapter/inr_data.py). No upstream provides
these files, so this script generates them once with a fixed-seed, per-class stratified 80/20 split;
provenance.json records seed / fraction / counts / sha1 (a self-generated seeded split).

Usage (run once after extracting the data):
  python make_inr_split.py                       # data/imagenet-r -> data/imagenet-r_split
  python make_inr_split.py --selftest            # self-test without data (temporary fake dir)

Safeguards: refuses to overwrite an existing split (--force overwrites; then also delete the
materialised data/imagenet-r-bilora directory); raises if the class count != 200
(--allow_nonstandard is for the self-test only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SEED = 1997
DEFAULT_TEST_FRAC = 0.2


def _sha1_file(p: Path):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_split(inr_root: Path, out_dir: Path, seed: int, test_frac: float,
                allow_nonstandard: bool, force: bool):
    if not inr_root.is_dir():
        raise FileNotFoundError(f"missing raw ImageNet-R directory: {inr_root}")
    train_txt, test_txt = out_dir / "train.txt", out_dir / "test.txt"
    if (train_txt.exists() or test_txt.exists()) and not force:
        raise RuntimeError(f"{out_dir} already has split files: they are frozen, refusing to overwrite silently. "
                           "To regenerate use --force and also delete the materialised data/imagenet-r-bilora/ directory.")

    classes = sorted(d.name for d in inr_root.iterdir() if d.is_dir())
    if len(classes) != 200 and not allow_nonstandard:
        raise RuntimeError(f"{inr_root} has {len(classes)} class dirs != 200: "
                           "not a complete ImageNet-R (or extra directories present); refusing to continue.")

    rng = np.random.RandomState(seed)
    train_lines, test_lines = [], []
    for wnid in classes:                       # fixed class order, per-class stratification
        files = sorted(p.name for p in (inr_root / wnid).iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if len(files) < 2:
            raise RuntimeError(f"class {wnid} has only {len(files)} images: incomplete data; refusing to continue.")
        perm = rng.permutation(len(files))
        n_test = max(1, int(round(test_frac * len(files))))
        te_idx = set(perm[:n_test].tolist())
        for i, fname in enumerate(files):
            (test_lines if i in te_idx else train_lines).append(f"{wnid}/{fname}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt.write_text("\n".join(train_lines) + "\n", encoding="utf-8", newline="\n")
    test_txt.write_text("\n".join(test_lines) + "\n", encoding="utf-8", newline="\n")
    prov = dict(source="make_inr_split.py (per-class stratified, frozen)",
                seed=seed, test_frac=test_frac, n_classes=len(classes),
                n_train=len(train_lines), n_test=len(test_lines),
                train_txt_sha1=_sha1_file(train_txt), test_txt_sha1=_sha1_file(test_txt))
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)
    print(f"[inr-split] written: {out_dir}  classes={len(classes)} "
          f"train={len(train_lines)} test={len(test_lines)} seed={seed} frac={test_frac}")
    print(f"[inr-split] sha1: train={prov['train_txt_sha1']} test={prov['test_txt_sha1']}")
    return prov


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "imagenet-r"
        for wnid, n in (("n001", 10), ("n002", 7), ("n003", 5)):
            d = root / wnid
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"img_{i:03d}.jpg").write_bytes(b"x")
        out = Path(td) / "imagenet-r_split"
        p1 = build_split(root, out, DEFAULT_SEED, DEFAULT_TEST_FRAC, True, False)
        assert p1["n_train"] + p1["n_test"] == 22 and p1["n_test"] == 2 + 1 + 1, p1
        lines = (out / "train.txt").read_text(encoding="utf-8").strip().splitlines()
        assert all(len(l.split("/")) == 2 for l in lines), "line format must be <wnid>/<file>"
        # determinism: regenerating with the same seed is byte-identical
        sha_before = p1["train_txt_sha1"]
        p2 = build_split(root, out, DEFAULT_SEED, DEFAULT_TEST_FRAC, True, True)
        assert p2["train_txt_sha1"] == sha_before, "same seed must reproduce byte-identically"
        # refuses silent overwrite
        try:
            build_split(root, out, DEFAULT_SEED, DEFAULT_TEST_FRAC, True, False)
            raise AssertionError("must raise when a split already exists")
        except RuntimeError:
            pass
    print("[inr-split] selftest OK (stratified counts / line format / same-seed determinism / refuses overwrite)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inr_root", default="data/imagenet-r")
    ap.add_argument("--out", default="data/imagenet-r_split")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--test_frac", type=float, default=DEFAULT_TEST_FRAC)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow_nonstandard", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    build_split(Path(args.inr_root), Path(args.out), args.seed, args.test_frac,
                args.allow_nonstandard, args.force)


if __name__ == "__main__":
    main()
