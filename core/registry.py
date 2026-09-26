"""Method registry + unified interface contract.

Every method module (baselines/bilora_adapter/<m>.py) exposes:

    def run(train_loaders, val_loaders, test_loaders, ncls, args, device,
            threshold=None) -> dict
        # returns the standard result dict:
        # {method, threshold, task_matrix, acc_curve, bwt_curve, cap_curve,
        #  selected_layers, layer_scores, runtime_sec}

    METHOD_SPEC = {"name": str, "needs_threshold": bool, "needs_selector": bool}

Add a new method = add a module + one line in REGISTRY. exp.py needs no edits.
"""
import importlib

REGISTRY = {
    # NOTE: 'bilora_adapter' (not 'bilora') avoids a case-insensitive-FS clash
    # with the upstream ./baselines/BiLoRA checkout on Windows.
    "bilora": "baselines.bilora_adapter.bilora",
    # BiLoRA + persistent shared adapter (shared/specific layer allocation host).
    "bilora_d2": "baselines.bilora_adapter.bilora_d2",
}


def get_method(name):
    if name not in REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Registered: {sorted(REGISTRY)}")
    return importlib.import_module(REGISTRY[name])
