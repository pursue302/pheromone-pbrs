"""Summarize a completed PointMaze experiment and plot original success rates."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "sarsa": "Unshaped SARSA",
    "return_all": "Return-D (all)",
    "fixed_mix": "Fixed mix",
    "return_gate": "Return-Gate",
}


def paired_difference(left, right, rng):
    """Estimate mean paired difference and a percentile bootstrap interval."""
    # The arrays align by seed. Resample the per-seed differences, so
    # the start cells within a trained policy stay in the same group.
    differences = left - right
    samples = rng.choice(differences, size=(20_000, len(differences)), replace=True)
    interval = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    # Under the paired null, swapping the methods within a seed only
    # changes the sign. Compare absolute means for a two-sided test.
    signs = rng.choice((-1, 1), size=samples.shape)
    null = (signs * differences).mean(axis=1)
    p_value = (1 + np.count_nonzero(abs(null) >= abs(differences.mean()) - 1e-12)) / (len(null) + 1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95": interval.tolist(),
        "paired_randomization_p": float(p_value),
        "wins_ties_losses": [
            int(np.sum(differences > 1e-12)),
            int(np.sum(abs(differences) <= 1e-12)),
            int(np.sum(differences < -1e-12)),
        ],
    }


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The run manifest must be complete.")
    methods = manifest["methods"]
    seeds = manifest["seeds"]
    data = {method: [] for method in methods}
    # Keep the manifest order so all method arrays align by training seed.
    for seed in seeds:
        rows = json.loads((folder / f"seed_{seed}.json").read_text())
        assert [row["method"] for row in rows] == methods
        assert all(row["seed"] == seed for row in rows)
        assert len({row["warmup_hash"] for row in rows}) == 1
        for row in rows:
            data[row["method"]].append(row)
    config = manifest["config"]
    steps = np.array([record["step"] for record in rows[0]["records"]])
    assert steps[-1] == config["steps"]
    summary = {"seed_count": len(seeds), "per_method": {}, "comparisons": {}}
    # Each method has the same warm-up trajectory for a given seed.
    # Count its successes once, using the unshaped arm as the reference.
    warmup = [row["warmup_successes"] for row in data["sarsa"]]
    summary["warmup"] = {
        "seeds_with_success": int(np.count_nonzero(warmup)),
        "successful_episodes_per_seed": warmup,
        "mean_steps": float(np.mean([row["warmup_steps"] for row in data["sarsa"]])),
    }
    summary["max_boundary_error"] = max(
        row["max_boundary_error"] for group in data.values() for row in group
    )
    metrics = {}
    fig, ax = plt.subplots(figsize=(7.8, 4.5), layout="constrained")
    for method, group in data.items():
        assert all([r["step"] for r in row["records"]] == steps.tolist() for row in group)
        # Rows are training seeds. Columns are evaluation checkpoints.
        # Each cell already averages success across the fixed start cells.
        curves = np.array([[r["success_rate"] for r in row["records"]] for row in group])
        mean_curve = curves.mean(axis=0)
        totals = [sum(length is not None for length in row["records"][-1]["steps_by_start"]) for row in group]
        # Integrate each curve over actual interaction counts, then divide
        # by the budget to keep the AUC on the same 0-to-1 scale.
        auc = np.trapezoid(curves, x=steps, axis=1) / steps[-1]
        # Report travel time conditional on success. The success rate above
        # still includes every timed-out rollout in its denominator.
        lengths = [
            length for row in group for length in row["records"][-1]["steps_by_start"]
            if length is not None
        ]
        summary["per_method"][method] = {
            "mean_success_rate_auc": float(auc.mean()),
            "final_success_rate": float(mean_curve[-1]),
            "final_successes": sum(totals),
            "final_evaluation_episodes": len(group) * len(manifest["evaluation_start_cells"]),
            "mean_steps_on_success": float(np.mean(lengths)) if lengths else None,
            "training_seeds_with_any_success": sum(row["first_training_success"] is not None for row in group),
        }
        metrics[method] = {"success_rate_auc": auc, "final_success_rate": curves[:, -1]}
        (line,) = ax.plot(steps / 1000, mean_curve, label=LABELS[method])
        # The shaded region is one standard error across training seeds.
        # Paired confidence intervals for differences are computed separately.
        if len(group) > 1:
            sem = curves.std(axis=0, ddof=1) / np.sqrt(len(group))
            ax.fill_between(steps / 1000, np.clip(mean_curve - sem, 0, 1),
                            np.clip(mean_curve + sem, 0, 1), color=line.get_color(), alpha=0.12)
    ax.axvline(summary["warmup"]["mean_steps"] / 1000, ls="--", lw=1, color="gray")
    ax.set(xlabel="Training interactions (thousands)", ylabel="Greedy evaluation success rate",
           title="MuJoCo PointMaze Medium", ylim=(-0.02, 1.02))
    ax.legend()
    fig.savefig(folder / "success_curves.png", dpi=160)
    plt.close(fig)

    # Seed the analysis independently of training for reproducible intervals.
    # The protocol treats only Gate versus Fixed mix AUC as the primary test.
    rng = np.random.default_rng(29)
    for baseline in methods:
        if baseline != "return_gate":
            summary["comparisons"][baseline] = {
                metric: paired_difference(scores, metrics[baseline][metric], rng)
                for metric, scores in metrics["return_gate"].items()
            }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run)
