# -*- coding: utf-8 -*-
"""CUB-200 for the BiLoRA host.

BiLoRA's iCUB (utils/data.py:66-67) ignores cfg["data_path"] and hard-codes data/cub/train/ and
data/cub/test/ relative to the current working directory, each an ImageFolder of 200 class dirs.
Two ways this can silently go wrong:
  - the working directory is not the repository root -> a different data/cub is read (or FileNotFoundError);
  - train/test class dirs differ -> ImageFolder numbers each split by its own sort order and labels
    are silently misaligned.
Before the host builds its DataManager this module checks: the relative path == <data_root>/cub; both
splits have the same 200 class dirs; image counts and the sha256 of the sorted relative file list match
the expected values. Any mismatch raises; nothing is patched or bypassed.
"""
from __future__ import annotations

import hashlib
import os

# Expected values for the APER CUB split (cub.zip; see docs/DATASETS.md).
EXPECT = {
    "train": (9430, "ff44d7e108094baccec4fb8daa1173dbb5b29a282cef81c79b7a2fb96f25f4d4"),
    "test": (2358, "07be292469c6135074cac429105c2e43ee76dbb551b3807a7a6bade98dc680d5"),
}
N_CLASSES = 200


def split_fingerprint(split_root):
    """Return (class dirs, file count, sha256 of the sorted relative file list); see docs/DATASETS.md."""
    classes = sorted(d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d)))
    files = sorted(os.path.relpath(os.path.join(dp, fn), split_root).replace(os.sep, "/")
                   for dp, _, fs in os.walk(split_root) for fn in fs)
    sha = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()
    return classes, len(files), sha


def ensure_bilora_cub(args, expect=EXPECT):
    """Return <data_root>/cub if all checks pass (BiLoRA cfg["data_path"]; iCUB actually reads data/cub under cwd)."""
    root = os.path.join(args.data_root, "cub")
    rel = os.path.join("data", "cub")  # where iCUB actually reads (relative to cwd)
    if not (os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "test"))):
        raise FileNotFoundError(f"missing CUB directory {root}/train or {root}/test (APER split, see docs/DATASETS.md)")
    if os.path.realpath(rel) != os.path.realpath(root):
        raise RuntimeError(f"BiLoRA iCUB reads {rel} under cwd (={os.path.realpath(rel)}), which is not the same place as "
                           f"{root} under data_root (={os.path.realpath(root)}) -> run from the repository root")
    got = {}
    for split in ("train", "test"):
        classes, n, sha = split_fingerprint(os.path.join(root, split))
        got[split] = (classes, n, sha)
        want_n, want_sha = expect[split]
        if len(classes) != N_CLASSES:
            raise RuntimeError(f"CUB {split} has {len(classes)} class dirs != {N_CLASSES}")
        if n != want_n or sha != want_sha:
            raise RuntimeError(f"CUB {split} has {n} files, file-list sha {sha[:16]}... != expected {want_n} / {want_sha[:16]}..."
                               " (split was modified, or it is not the APER version)")
    if got["train"][0] != got["test"][0]:
        raise RuntimeError("CUB train/test class dirs differ -> ImageFolder labels would be misaligned; refusing to continue")
    print(f"[d2] cub dir verified: {root} (APER split, train={got['train'][1]} test={got['test'][1]} "
          f"classes={N_CLASSES}, filelist={got['train'][2][:16]}/{got['test'][2][:16]})")
    return root
