"""Dataset / transform logic. CIFAR-100 transforms are aligned to the BiLoRA setup."""
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# Aligned with BiLoRA (utils/data.py, iCIFAR100): raw-pixel Normalize(mean=0, std=1);
# train augmentation RandomResizedCrop(224)+HFlip, test Resize(224) only. Normalize(0,1)
# is the identity and is kept only to match BiLoRA exactly.
_NORM_MEAN = (0.0, 0.0, 0.0)
_NORM_STD = (1.0, 1.0, 1.0)


def tfm_train():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # defaults scale=(0.08,1.0), ratio=(3/4,4/3), as in iCIFAR100.train_trsf
            transforms.RandomHorizontalFlip(),  # default p=0.5
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ]
    )


def tfm_test():
    return transforms.Compose(
        [
            transforms.Resize(224),  # shorter-side resize; CIFAR 32x32 squares -> 224x224, same as Resize((224,224))
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ]
    )


# ImageNet normalization for ImageNet-R (iBOT backbone). This deliberately departs from the
# CIFAR Normalize(0,1) quirk above: the backbone was pretrained on ImageNet-normalized inputs,
# and raw pixels would cause a distribution mismatch and weaker features.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def tfm_train_inr():
    # Standard ImageNet train augmentation, RandomResizedCrop(224)+HFlip; ImageNet normalization (see above).
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # defaults scale=(0.08,1.0), ratio=(3/4,4/3)
            transforms.RandomHorizontalFlip(),  # default p=0.5
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def tfm_test_inr():
    # Standard ImageNet eval preprocessing, Resize(256)+CenterCrop(224); ImageNet-R images are full-size, so the CIFAR Resize(224) is not reused.
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def build_cifar100_split(args):
    # Train subsets use augmentation (tfm_train); val/test subsets use none (tfm_test).
    # tr_aug and tr_eval are the same training data with different transforms.
    tr_aug = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=tfm_train())  # train (augmented)
    tr_eval = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=tfm_test())  # train (not augmented, for val)
    te = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tfm_test())  # test
    total_classes = len(tr_aug.classes)
    if args.num_tasks * args.classes_per_task > total_classes:  # num_tasks * classes_per_task must not exceed the class count
        raise ValueError("num_tasks * classes_per_task exceeds CIFAR-100 classes")

    if args.task_order == "chrono":  # default order
        order = list(range(total_classes))
    else:
        # Same as BiLoRA data_manager._setup_data: class order from np.random.permutation(seed), so the
        # class-to-task grouping matches BiLoRA shuffle=True (our shared head uses global labels, no remapping).
        order = np.random.RandomState(args.seed).permutation(total_classes).tolist()  # optional shuffled order

    task_classes = [order[i * args.classes_per_task : (i + 1) * args.classes_per_task] for i in range(args.num_tasks)]  # global class ids of each task (nested list)
    tr_targets = np.array(tr_aug.targets)  # numpy arrays for np.isin indexing
    te_targets = np.array(te.targets)
    train_loaders, val_loaders, test_loaders = [], [], []
    pin_memory = torch.cuda.is_available()  # pinned host memory for faster host-to-GPU copies

    for tid, cls in enumerate(task_classes):  # cls is a list of class ids
        tr_idx = np.where(np.isin(tr_targets, cls))[0]  # indices of this task's images
        te_idx = np.where(np.isin(te_targets, cls))[0]
        rng = np.random.RandomState(args.seed + tid)
        tr_idx = rng.permutation(tr_idx)  # shuffle with seed + tid to avoid a fixed-order bias
        nval = int(round(len(tr_idx) * args.val_split))  # validation size; split by slicing
        val_idx = tr_idx[:nval].tolist()  # numpy -> list
        train_idx = tr_idx[nval:].tolist()

        # Subset keeps the original CIFAR-100 labels, so y is always a global class id (0-99);
        # no per-task remapping, as required for the class-incremental label space.
        train_subset = Subset(tr_aug, train_idx)  # train subset: augmented
        val_subset = Subset(tr_eval, val_idx) if val_idx else Subset(tr_eval, [])  # val subset: not augmented
        test_subset = Subset(te, te_idx.tolist())

        train_loaders.append(  # build loaders
            DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        )
        val_loaders.append(
            DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        )
        test_loaders.append(
            DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        )
    return train_loaders, val_loaders, test_loaders, task_classes, total_classes


def _imagenet_r_train_test_split(base, args):
    """Return (train_idx, test_idx) for ImageNet-R, which has no built-in train/test split.

    Uses the official split files <data_root>/imagenet-r_split/{train.txt,test.txt} (lists of
    relative image paths) when present, for a paper-comparable split; otherwise falls back to
    a per-class split seeded by args.seed. Indices refer to ImageFolder.samples; labels stay
    global class ids.
    """
    split_dir = os.path.join(args.data_root, "imagenet-r_split")
    train_txt = os.path.join(split_dir, "train.txt")
    test_txt = os.path.join(split_dir, "test.txt")
    # ImageFolder stores each image's absolute path in samples[i][0]; map relative paths back to row indices.
    inr_root = os.path.join(args.data_root, "imagenet-r")
    if os.path.isfile(train_txt) and os.path.isfile(test_txt):
        # Paper-comparable path: follow the official files exactly, no randomness.
        print("[build_imagenet_r_split] using OFFICIAL split files at %s (paper-comparable)" % split_dir)
        rel2idx = {}  # relative path (forward slashes) -> ImageFolder row index
        for i, (path, _y) in enumerate(base.samples):
            rel = os.path.relpath(path, inr_root).replace(os.sep, "/")
            rel2idx[rel] = i

        def _read(txt):
            idxs = []
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    rel = line.strip().replace("\\", "/")
                    if not rel:
                        continue
                    if rel not in rel2idx:  # fail early if the split file and disk disagree, instead of silently dropping samples
                        raise ValueError("split entry not found under %s: %s" % (inr_root, rel))
                    idxs.append(rel2idx[rel])
            return np.array(sorted(idxs))

        return _read(train_txt), _read(test_txt)

    # Fallback: no official files; deterministic per-class split by seed, with a clear warning that it is not paper-comparable.
    test_frac = getattr(args, "imagenet_r_test_frac", 0.2)
    print(
        "[build_imagenet_r_split] WARNING: no official split at %s; "
        "falling back to seeded per-class split (seed=%d, test_frac=%.3f) — NOT paper-comparable"
        % (split_dir, args.seed, test_frac)
    )
    targets = np.array([y for _p, y in base.samples])  # global class id of every sample
    train_idx, test_idx = [], []
    for c in np.unique(targets):  # split per class so every class has the same train/test ratio
        cls_idx = np.where(targets == c)[0]
        rng = np.random.RandomState(args.seed + int(c))  # seed includes the class id: independent and reproducible per class
        cls_idx = rng.permutation(cls_idx)
        ntest = int(round(len(cls_idx) * test_frac))
        test_idx.extend(cls_idx[:ntest].tolist())
        train_idx.extend(cls_idx[ntest:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def build_imagenet_r_split(args):
    """Class-incremental task split for ImageNet-R, same contract as build_cifar100_split:
    returns (train_loaders, val_loaders, test_loaders, task_classes, total_classes).

    ImageNet-R has 200 ImageNet classes (one folder per wnid, ~30k images) and no built-in
    train/test split, so one is made by _imagenet_r_train_test_split (official split files
    if present, otherwise a seeded per-class split).
    """
    inr_root = os.path.join(args.data_root, "imagenet-r")
    # One copy with augmentation, one without; both scan the same root, so sample order and indices match (like CIFAR's tr_aug/tr_eval).
    tr_aug = torchvision.datasets.ImageFolder(root=inr_root, transform=tfm_train_inr())  # train candidates (augmented)
    tr_eval = torchvision.datasets.ImageFolder(root=inr_root, transform=tfm_test_inr())  # same images, not augmented (for val/test)
    total_classes = len(tr_aug.classes)  # always 200 for ImageNet-R
    if args.num_tasks * args.classes_per_task > total_classes:  # num_tasks * classes_per_task must not exceed the class count
        raise ValueError("num_tasks * classes_per_task exceeds ImageNet-R classes")

    # No built-in train/test: first make a global train/test index split, then assign classes to tasks.
    all_train_idx, all_test_idx = _imagenet_r_train_test_split(tr_aug, args)
    targets = np.array([y for _p, y in tr_aug.samples])  # global class id of every sample
    train_mask = np.zeros(len(targets), dtype=bool)
    train_mask[all_train_idx] = True  # boolean mask, ANDed with "class belongs to this task" below

    if args.task_order == "chrono":  # default order
        order = list(range(total_classes))
    else:
        # Same as the CIFAR branch: np.random.permutation(seed) decides which classes go to which task; global labels, no remapping.
        order = np.random.RandomState(args.seed).permutation(total_classes).tolist()  # optional shuffled order

    task_classes = [order[i * args.classes_per_task : (i + 1) * args.classes_per_task] for i in range(args.num_tasks)]  # global class ids of each task
    train_loaders, val_loaders, test_loaders = [], [], []
    pin_memory = torch.cuda.is_available()  # pinned host memory for faster host-to-GPU copies

    for tid, cls in enumerate(task_classes):  # cls is a list of class ids
        cls_mask = np.isin(targets, cls)  # samples of this task's classes
        tr_idx = np.where(cls_mask & train_mask)[0]  # this task's train samples (global indices)
        te_idx = np.where(cls_mask & ~train_mask)[0]  # this task's test samples
        rng = np.random.RandomState(args.seed + tid)
        tr_idx = rng.permutation(tr_idx)  # shuffle with seed + tid to avoid a fixed-order bias
        nval = int(round(len(tr_idx) * args.val_split))  # carve val out of train, as for CIFAR
        val_idx = tr_idx[:nval].tolist()
        train_idx = tr_idx[nval:].tolist()

        # Subset keeps the ImageFolder labels, so y is always a global class id (0-199); no per-task remapping (class-incremental).
        train_subset = Subset(tr_aug, train_idx)  # train subset: augmented
        val_subset = Subset(tr_eval, val_idx) if val_idx else Subset(tr_eval, [])  # val subset: not augmented
        test_subset = Subset(tr_eval, te_idx.tolist())  # test subset: not augmented (reuses tr_eval instead of another dataset)

        train_loaders.append(  # build loaders
            DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        )
        val_loaders.append(
            DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        )
        test_loaders.append(
            DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        )
    return train_loaders, val_loaders, test_loaders, task_classes, total_classes


def build_cub_split(args):
    """Class-incremental task split for CUB-200 (APER layout: <data_root>/cub/{train,test}/<class dir>/),
    same contract as above.

    The D2 bilora host trains and reads out through BiLoRA's own DataManager (iCUB) and does not
    use these loaders; this function keeps exp.py's generic entry from silently falling back to
    CIFAR-100 for cub, and gives other methods a split with the host's class order
    (chrono = lexicographic class-directory order)."""
    root = os.path.join(args.data_root, "cub")
    tr_aug = torchvision.datasets.ImageFolder(root=os.path.join(root, "train"), transform=tfm_train_inr())
    tr_eval = torchvision.datasets.ImageFolder(root=os.path.join(root, "train"), transform=tfm_test_inr())
    te = torchvision.datasets.ImageFolder(root=os.path.join(root, "test"), transform=tfm_test_inr())
    if tr_aug.classes != te.classes:  # mismatched class dirs -> ImageFolder numbers each side separately, labels silently shift
        raise ValueError("CUB train/test class directories differ; refusing to continue")
    total_classes = len(tr_aug.classes)
    if args.num_tasks * args.classes_per_task > total_classes:
        raise ValueError("num_tasks * classes_per_task exceeds CUB classes")
    if args.task_order == "chrono":
        order = list(range(total_classes))
    else:
        order = np.random.RandomState(args.seed).permutation(total_classes).tolist()
    task_classes = [order[i * args.classes_per_task : (i + 1) * args.classes_per_task] for i in range(args.num_tasks)]
    tr_targets = np.array(tr_aug.targets)
    te_targets = np.array(te.targets)
    train_loaders, val_loaders, test_loaders = [], [], []
    pin_memory = torch.cuda.is_available()
    for tid, cls in enumerate(task_classes):
        tr_idx = np.where(np.isin(tr_targets, cls))[0]
        te_idx = np.where(np.isin(te_targets, cls))[0]
        rng = np.random.RandomState(args.seed + tid)
        tr_idx = rng.permutation(tr_idx)
        nval = int(round(len(tr_idx) * args.val_split))
        val_idx = tr_idx[:nval].tolist()
        train_idx = tr_idx[nval:].tolist()
        train_loaders.append(
            DataLoader(Subset(tr_aug, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        )
        val_loaders.append(
            DataLoader(Subset(tr_eval, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        )
        test_loaders.append(
            DataLoader(Subset(te, te_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        )
    return train_loaders, val_loaders, test_loaders, task_classes, total_classes


def build_split(args):
    """Dataset dispatcher that keeps the entry point dataset-agnostic; dispatches on
    args.dataset (default cifar100). Unknown values raise: silently falling back to
    CIFAR-100 would let a misspelled or unwired dataset name run and report numbers
    that are really CIFAR-100."""
    dataset = getattr(args, "dataset", "cifar100")  # set by --dataset; read-only here, default cifar100
    if dataset == "imagenet_r":
        return build_imagenet_r_split(args)
    if dataset == "cub":
        return build_cub_split(args)
    if dataset == "cifar100":
        return build_cifar100_split(args)
    raise ValueError(f"build_split: unknown dataset={dataset!r} (choices: cifar100/imagenet_r/cub)")


# Cache of non-augmented training-split datasets (test transform): NCM prototypes should be computed
# on clean features matching the test distribution; the RandomResizedCrop augmentation of
# train_loaders would misalign prototypes and lower task-routing accuracy.
_EVAL_BASE_CACHE = {}


def proto_loader_from(train_loader, args):
    """Return a DataLoader over the **same samples** as train_loader but with the
    non-augmented test transform, used to compute NCM routing prototypes that match the
    test distribution.

    Relies on the build_*_split contract: train_loader.dataset is Subset(<train dataset>, indices).
    """
    subset = train_loader.dataset
    if not hasattr(subset, "indices"):
        # Fallback for non-Subset datasets (test stubs): nothing to re-wrap, so return an unshuffled
        # loader over the same dataset. The real pipeline always takes the Subset path below.
        return DataLoader(
            subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
    dataset = getattr(args, "dataset", "cifar100")  # read the same way as build_split
    cache_key = (args.data_root, dataset)  # keyed by (root, dataset) so caches of different datasets never mix
    if cache_key not in _EVAL_BASE_CACHE:
        if dataset == "imagenet_r":
            # ImageNet-R non-augmented base: ImageFolder + tfm_test_inr(); sample order matches tr_aug in build_imagenet_r_split, so indices carry over.
            _EVAL_BASE_CACHE[cache_key] = torchvision.datasets.ImageFolder(
                root=os.path.join(args.data_root, "imagenet-r"), transform=tfm_test_inr()
            )
        elif dataset == "cub":
            # CUB non-augmented base: same root and scan as tr_aug in build_cub_split, so indices carry over.
            _EVAL_BASE_CACHE[cache_key] = torchvision.datasets.ImageFolder(
                root=os.path.join(args.data_root, "cub", "train"), transform=tfm_test_inr()
            )
        elif dataset != "cifar100":
            raise ValueError(f"proto_loader_from: unknown dataset={dataset!r} (choices: cifar100/imagenet_r/cub)")
        else:
            # download=False: build_cifar100_split has already run and the data is on disk; reuse it.
            _EVAL_BASE_CACHE[cache_key] = torchvision.datasets.CIFAR100(
                root=args.data_root, train=True, download=False, transform=tfm_test()
            )
    eval_subset = Subset(_EVAL_BASE_CACHE[cache_key], subset.indices)
    return DataLoader(
        eval_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
