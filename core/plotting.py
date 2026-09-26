"""Result plotting helpers."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(results, out_dir):
    plt.figure(figsize=(8, 5))
    for result in results:
        x = np.arange(1, len(result["acc_curve"]) + 1)
        label = result["method"] if result["threshold"] is None else f"{result['method']}(th={result['threshold']})"
        plt.plot(x, result["acc_curve"], marker="o", label=label)
    plt.xlabel("Increasing No. of Tasks")
    plt.ylabel("Average Accuracy (ACC)")
    plt.title("ACC vs Tasks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_tasks.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for result in results:
        x = np.arange(1, len(result["cap_curve"]) + 1)
        label = result["method"] if result["threshold"] is None else f"{result['method']}(th={result['threshold']})"
        plt.plot(x, result["cap_curve"], marker="o", label=label)
    plt.xlabel("Task ID")
    plt.ylabel("CAP")
    plt.title("Capacity vs Tasks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "capacity_vs_tasks.png", dpi=200)
    plt.close()

    # Threshold-ablation and layer-selection plots shared by the model_m method family.
    # "model_m" keeps the unsuffixed file names; the others get a method suffix so they do not overwrite.
    for method_name in ("model_m", "model_m_2", "model_m_timm", "model_m_timm_ncm",
                        "model_m_timm_v31", "model_m_timm_v31_mlp"):
        suffix = "" if method_name == "model_m" else f"_{method_name}"
        mr = sorted(
            [r for r in results if r["method"] == method_name],
            key=lambda d: d["threshold"] if d["threshold"] is not None else -1,
        )
        if not mr:
            continue
        th = [r["threshold"] for r in mr]
        facc = [r["acc_curve"][-1] for r in mr]
        fbwt = [r["bwt_curve"][-1] for r in mr]
        plt.figure(figsize=(8, 5))
        plt.plot(th, facc, marker="o", label="Final ACC")
        plt.plot(th, fbwt, marker="s", label="Final BWT")
        plt.xlabel("threshold")
        plt.ylabel("Score")
        plt.title(f"Threshold Ablation ({method_name})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"threshold_ablation{suffix}.png", dpi=200)
        plt.close()

        best = max(mr, key=lambda d: d["acc_curve"][-1])
        sm = np.array(best["layer_scores"], dtype=np.float32)
        bm = np.zeros_like(sm)
        for t, layers in enumerate(best["selected_layers"]):
            for layer in layers:
                if 0 <= layer < bm.shape[1]:
                    bm[t, layer] = 1.0
        plt.figure(figsize=(10, 4))
        plt.imshow(sm, aspect="auto", cmap="viridis")
        plt.colorbar(label="Selection Score")
        plt.xlabel("Layer")
        plt.ylabel("Task")
        plt.title(f"Layer Sensitivity ({method_name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"layer_sensitivity_scores{suffix}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.imshow(bm, aspect="auto", cmap="Greys")
        plt.xlabel("Layer")
        plt.ylabel("Task")
        plt.title(f"Layer Selection Matrix ({method_name})")
        plt.tight_layout()
        plt.savefig(out_dir / f"layer_selection_matrix{suffix}.png", dpi=200)
        plt.close()
