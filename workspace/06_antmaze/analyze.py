"""Summarize paired AntMaze results by training seed and start difficulty."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "sac": "Sparse SAC",
    "return_all": "Return-D (all)",
    "fixed_mix": "Fixed mix",
    "return_gate": "Return-Gate",
}


def compare(left, right, rng):
    differences = left - right
    bootstrap = rng.choice(differences, size=(20_000, len(differences)), replace=True)
    signs = rng.choice((-1, 1), size=bootstrap.shape)
    null = np.abs((signs * differences).mean(axis=1))
    return {
        "difference": float(differences.mean()),
        "ci95": np.quantile(bootstrap.mean(axis=1), [0.025, 0.975]).tolist(),
        "paired_p": float((1 + np.sum(null >= abs(differences.mean()) - 1e-12)) / (len(null) + 1)),
    }


def start_rate(rows, tier):
    values = [value for row in rows for value in row["steps_by_start"][tier]]
    return float(np.mean([value is not None for value in values]))


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Analyze a completed run.")
    seeds = [json.loads((folder / f"seed_{seed}.json").read_text()) for seed in manifest["seeds"]]
    prefix = manifest["config"]["warmup_steps"]
    warmup = [
        dict(row["selected_checkpoint"]["evaluation"], step=prefix)
        if row.get("selected_checkpoint") else row["warmup"]["records"][-1]
        for row in seeds
    ]
    summary = {
        "seed_count": len(seeds),
        "warmup_evaluation_success": float(np.mean([row["success_rate"] for row in warmup])),
        "warmup_success_by_start": {tier: start_rate(warmup, tier) for tier in manifest["start_cells"]},
        "warmup_successful_trajectories": [row["warmup"]["training_successes"] for row in seeds],
    }
    if "common_checkpoint" in manifest:
        summary["common_checkpoint"] = manifest["common_checkpoint"]
        summary["selected_checkpoint_steps"] = [
            row["selected_checkpoint"]["training_step"] if row.get("selected_checkpoint") else None
            for row in seeds
        ]
    if not manifest["pilot"]:
        summary.update(compare_methods(folder, manifest, seeds, warmup))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def compare_methods(folder, manifest, seeds, warmup):
    prefix = manifest["config"]["warmup_steps"]
    per_method, metrics = {}, {}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    warmup_records = [row["warmup"]["records"] for row in seeds]
    warmup_steps = [item["step"] for item in warmup_records[0]]
    axes[0].scatter(prefix / 1000, np.mean([row["success_rate"] for row in warmup]),
                    color="black", marker="D", label="Selected checkpoints")
    axes[0].plot(np.array(warmup_steps) / 1000,
                 np.mean([[item["success_rate"] for item in rows] for rows in warmup_records], axis=0),
                 color="gray", ls="--", label="Shared warm start")
    rng = np.random.default_rng(93)
    for method in manifest["methods"]:
        groups = [row["methods"][method] for row in seeds]
        records = [row["records"] for row in groups]
        assert [rows[0] for rows in records] == warmup
        assert max(row["max_telescoping_error"] for row in groups) < 1e-6
        steps = np.array([item["step"] for item in records[0]])
        curves = np.array([[item["success_rate"] for item in rows] for rows in records])
        # Measure sample efficiency after the split. The training prefix
        # contributes equally to all methods and precedes the experiment.
        auc = np.trapezoid(curves, x=steps, axis=1) / (steps[-1] - prefix)
        final = [rows[-1] for rows in records]
        discounted = np.array([row["discounted_return"] for row in final])
        per_method[method] = {
            "post_warmup_success_auc": float(auc.mean()),
            "final_success": float(curves[:, -1].mean()),
            "final_discounted_return": float(discounted.mean()),
            "final_success_by_start": {tier: start_rate(final, tier) for tier in manifest["start_cells"]},
            "training_successes": int(sum(row["training_successes"] for row in groups)),
        }
        metrics[method] = (auc, curves[:, -1])
        (line,) = axes[0].plot(steps / 1000, curves.mean(axis=0), label=LABELS[method])
        if len(seeds) > 1:
            sem = curves.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            axes[0].fill_between(steps / 1000, np.maximum(0, curves.mean(axis=0) - sem),
                                 np.minimum(1, curves.mean(axis=0) + sem),
                                 color=line.get_color(), alpha=0.10)
        tiers = per_method[method]["final_success_by_start"]
        axes[1].plot(list(tiers), list(tiers.values()), "-o", color=line.get_color(), label=LABELS[method])
    axes[0].axvline(prefix / 1000, color="gray", linewidth=1)
    axes[0].set(xlabel="Training interactions (thousands)", ylabel="Evaluation success rate",
                title="Learning curves", ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Start tier", ylabel="Final success rate", title="By start distance", ylim=(-0.02, 1.02))
    axes[0].legend()
    figure.savefig(folder / "success_curves.png", dpi=160)
    plt.close(figure)

    paired = {}
    for method, (auc, final) in metrics.items():
        if method != "return_gate":
            paired[method] = {
                "post_warmup_success_auc": compare(metrics["return_gate"][0], auc, rng),
                "final_success": compare(metrics["return_gate"][1], final, rng),
            }
    return {"per_method": per_method, "return_gate_minus": paired}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run)
