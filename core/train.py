"""Training / optimization helpers. Cosine LR schedule matches BiLoRA."""
import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(params, args, lr, weight_decay=None, optimizer_name=None):
    optimizer_name = optimizer_name or args.optimizer
    weight_decay = args.weight_decay if weight_decay is None else weight_decay
    if optimizer_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def cosine_lr_factor(ep, epochs):
    """BiLoRA CosineSchedule factor (see train_model): 1.0 for the first two epochs, then cosine decay.
    For custom training loops that scale a base lr per epoch, consistent with train_model."""
    if epochs <= 1:
        return 1.0
    return math.cos((99 * math.pi * max(ep - 1, 0)) / (200 * (epochs - 1)))


def build_param_groups(model, *, head_lr, adapter_lr, default_lr):
    """Split trainable params into head (classifier) / adapter (LoRA) / other groups with their own base lr.

    Reproduces BiLoRA's per-group learning rates (large for the head, small for adapters).
    head_lr/adapter_lr of None fall back to default_lr. Returns [{'params': [...], 'lr': base}, ...]
    where base is the pre-schedule lr each group is scaled from. Only requires_grad=True params."""
    head_params, adapter_params, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname:
            adapter_params.append(p)
        elif "shared_head" in lname or "head_pool" in lname or lname.endswith(".fc.weight") or lname.endswith(".fc.bias") or ".head." in lname:
            head_params.append(p)
        else:
            other.append(p)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr if head_lr else default_lr})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": adapter_lr if adapter_lr else default_lr})
    if other:
        groups.append({"params": other, "lr": default_lr})
    if not groups:
        raise RuntimeError("build_param_groups: no trainable params (nothing unfrozen?)")
    return groups


def forward_logits(model, x, task_id=None, restrict_to_seen=None):
    if getattr(model, "accepts_task_id", False):
        kwargs = {}
        if task_id is not None:
            kwargs["task_id"] = task_id
        if restrict_to_seen is not None:
            kwargs["restrict_to_seen"] = restrict_to_seen
        return model(x, **kwargs)
    if getattr(model, "task_aware", False):  # multi-head models need the task id
        if task_id is None:
            raise ValueError("task_id is required for task-aware models")
        return model(x, task_id)
    # Only models that support seen-class masking (e.g. SharedHeadCLIPClassifier) get restrict_to_seen;
    # CLIPClassifier (legacy / selector_model) does not accept it and is called as model(x).
    if restrict_to_seen is not None and hasattr(model, "mask_unseen_logits"):
        return model(x, restrict_to_seen=restrict_to_seen)
    return model(x)

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    args,
    *,
    epochs,
    lr,
    task_id=None,
    reg_fn=None,  # optional regularization term added to the loss
    max_steps=0,
    optimizer_name=None,
):
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable params")
    opt = build_optimizer(params, args, lr, optimizer_name=optimizer_name)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "amp" and device.type == "cuda")
    best_state = None
    best_val_loss = float("inf")
    patience = args.lr_patience
    current_lr = lr
    # Exact copy of BiLoRA's CosineSchedule (utils/schedulers.py + _LRScheduler): its __init__
    # resets last_epoch to -1 after the initial step() and steps only at epoch end, so epochs 0 and 1
    # both use base_lr. Epoch ep (0-indexed) therefore uses index max(ep-1, 0):
    # lr = base_lr * cos(99*pi*max(ep-1,0) / (200*(K-1))), K = epochs.
    use_cosine = getattr(args, "lr_schedule", "constant") == "cosine"

    for ep in range(epochs):
        if use_cosine:
            cur = lr * math.cos((99 * math.pi * max(ep - 1, 0)) / (200 * (epochs - 1))) if epochs > 1 else lr
            for group in opt.param_groups:
                group["lr"] = cur
        model.train()
        total = correct = 0
        run_loss = 0.0
        n = 0
        stop_early = False
        for i, (x, y) in enumerate(train_loader, start=1):
            if max_steps and i > max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.precision == "amp" and device.type == "cuda"):
                logits = forward_logits(model, x, task_id, restrict_to_seen=False)
                loss = loss_fn(logits, y)
                if reg_fn is not None:
                    loss = loss + reg_fn()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            run_loss += loss.item()
            n += 1

        train_loss = run_loss / max(n, 1)
        train_acc = correct / max(total, 1)
        line = f"  Epoch[{ep + 1}/{epochs}] loss={train_loss:.4f} train_acc={train_acc:.4f}"
        if len(val_loader.dataset) > 0:
            val_acc, val_loss = eval_model(
                model,
                val_loader,
                device,
                args.max_eval_steps,
                task_id=task_id,
                return_loss=True,
                restrict_to_seen=False,
            )
            line += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            if val_loss < best_val_loss - 1e-8:  # new best checkpoint
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = args.lr_patience
                line += " *"
            elif args.lr_patience > 0:
                patience -= 1
                if patience <= 0:
                    current_lr /= args.lr_factor
                    for group in opt.param_groups:
                        group["lr"] = current_lr
                    patience = args.lr_patience
                    line += f" lr={current_lr:.1e}"
                    if current_lr < args.lr_min:
                        stop_early = True
        print(line)
        if stop_early:
            break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def eval_model(model, loader, device, max_steps=0, task_id=None, return_loss=False, restrict_to_seen=None, return_stats=False):
    model.to(device).eval()
    total = correct = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader, start=1):
        if max_steps and i > max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = forward_logits(model, x, task_id, restrict_to_seen=restrict_to_seen)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        total_loss += loss_fn(logits, y).item() * y.size(0)
    acc = correct / max(total, 1)
    loss = total_loss / max(total, 1)
    if return_loss and return_stats:
        return acc, loss, total
    if return_loss:
        return acc, loss
    if return_stats:
        return acc, total
    return acc
