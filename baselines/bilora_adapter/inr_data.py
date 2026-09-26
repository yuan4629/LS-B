# -*- coding: utf-8 -*-
"""ImageNet-R for the BiLoRA host: materialise this repository's official split as the train/test
directories BiLoRA expects.

Conventions on the two sides:
  - this repository (core/data.py): <data_root>/imagenet-r/ is the raw 200-class ImageFolder;
    the official split is <data_root>/imagenet-r_split/{train.txt,test.txt} (relative path lists).
  - BiLoRA (utils/data.py, iIMAGENET_R): data_path must contain train/ and test/ ImageFolders.
    If they are missing, BiLoRA performs an unseeded 80/20 random_split and moves files into them;
    that split differs from the one used by the other methods and is not reproducible, so it must
    never be triggered.

This module materialises the official split under <data_root>/imagenet-r-bilora/{train,test}/<wnid>/
(hard links preferred, symlinks as fallback; both byte-identical to the originals) and writes
provenance.json with source and counts; later calls verify and reuse it. Every failure path raises
(missing split, directory of unknown origin, links not possible); there is no fallback to BiLoRA's
own split.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

_PROVENANCE = "provenance.json"


def _sha1_file(p: Path):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_split(txt: Path):
    rels = []
    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip().replace("\\", "/")
            if rel:
                rels.append(rel)
    if not rels:
        raise RuntimeError(f"{txt} is empty: the official split file is broken; refusing to continue.")
    return rels


def _link(src: Path, dst: Path):
    """Hard link first (zero-copy on the same volume, no privileges needed); symlink on EXDEV or when
    unsupported; otherwise raise. Both success paths are byte-identical; a silent copy fallback would
    consume several GB without trace, so failure raises instead."""
    try:
        os.link(str(src), str(dst))
        return
    except OSError:
        pass
    try:
        os.symlink(str(src), str(dst))
        return
    except OSError as e:
        raise RuntimeError(
            f"could create neither a hard link nor a symlink: {src} -> {dst} ({e}). "
            "Run on a filesystem that supports links, or materialise the directory manually and retry.")


def _materialize(inr_root: Path, rels, dst_split: Path, wnids):
    dst_split.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel in rels:
        src = inr_root / rel
        if not src.is_file():
            raise FileNotFoundError(f"image referenced by the official split does not exist: {src} (split and data do not match; refusing to continue)")
        parts = Path(rel).parts
        if len(parts) != 2:
            raise RuntimeError(f"split line is not of the form <wnid>/<file>: {rel!r}")
        wnids.add(parts[0])
        cls_dir = dst_split / parts[0]
        cls_dir.mkdir(exist_ok=True)
        _link(src, cls_dir / parts[1])
        n += 1
    return n


def ensure_bilora_imagenet_r(args):
    """Return the directory BiLoRA's cfg["data_path"] should point to (contains train/ test/). Idempotent; verifies provenance."""
    root = Path(args.data_root)
    inr_root = root / "imagenet-r"
    split_dir = root / "imagenet-r_split"
    train_txt, test_txt = split_dir / "train.txt", split_dir / "test.txt"
    dst = root / "imagenet-r-bilora"
    marker = dst / _PROVENANCE

    if dst.exists():
        if not marker.is_file():
            raise RuntimeError(
                f"{dst} exists but has no {_PROVENANCE}: unknown origin (possibly BiLoRA's own unseeded split). "
                "Check it manually, delete the directory and rerun; refusing to train on a split of unknown origin.")
        with open(marker, encoding="utf-8") as f:
            prov = json.load(f)
        for split, key in (("train", "n_train"), ("test", "n_test")):
            got = sum(1 for _ in (dst / split).rglob("*") if _.is_file())
            if got != int(prov[key]):
                raise RuntimeError(
                    f"{dst}/{split} has {got} files != {prov[key]} recorded in provenance: "
                    "the directory was modified; delete it and rerun to rebuild.")
        print(f"[d2] imagenet-r bilora dir reused: {dst} (official split, "
              f"train={prov['n_train']} test={prov['n_test']} classes={prov['n_classes']}, "
              f"src={prov['train_txt_sha1']}/{prov['test_txt_sha1']})")
        return str(dst)

    if not inr_root.is_dir():
        raise FileNotFoundError(f"missing raw ImageNet-R directory: {inr_root} (the standard data path is used; no other locations are tried).")
    if not (train_txt.is_file() and test_txt.is_file()):
        raise FileNotFoundError(
            f"missing official split files {train_txt} / {test_txt}. Refusing to fall back to BiLoRA's unseeded 80/20 split "
            "(it differs from the official split used by the other methods and is not reproducible).")

    building = root / "imagenet-r-bilora.building"
    if building.exists():
        shutil.rmtree(building)          # leftover from an interrupted build; rebuild
    wnids = set()
    tr = _materialize(inr_root, _read_split(train_txt), building / "train", wnids)
    te = _materialize(inr_root, _read_split(test_txt), building / "test", wnids)
    if len(wnids) != 200:
        raise RuntimeError(f"split covers {len(wnids)} classes != 200: the official split files are incomplete; refusing to continue.")
    prov = dict(source="official imagenet-r_split", n_train=tr, n_test=te, n_classes=len(wnids),
                train_txt_sha1=_sha1_file(train_txt), test_txt_sha1=_sha1_file(test_txt))
    with open(building / _PROVENANCE, "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)
    os.rename(building, dst)             # atomic: either a complete directory or a .building leftover cleaned up next time
    print(f"[d2] imagenet-r bilora dir materialised: {dst} (official split, "
          f"train={tr} test={te} classes={len(wnids)})")
    return str(dst)
