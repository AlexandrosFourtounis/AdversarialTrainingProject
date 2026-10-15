"""Compare two analysis JSON files and export a figure and Markdown summary."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compare_results(standard_path, robust_path, output_dir):
    """Plot recorded metrics without running training or evaluating checkpoints."""
    standard = json.loads(Path(standard_path).read_text())
    robust = json.loads(Path(robust_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    standard_stats = standard["attack_success_analysis"]
    robust_stats = robust["attack_success_analysis"]
    snapshots = [standard, robust]
    historical = any(
        item.get("evaluation", {}).get("attack_space") != "pixel" for item in snapshots
    )
    context = "Historical project results" if historical else "Evaluation results"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    positions = np.arange(2)
    width = 0.35
    clean = [stats["clean_accuracy"] for stats in (standard_stats, robust_stats)]
    adversarial = [stats["adv_accuracy"] for stats in (standard_stats, robust_stats)]
    axes[0].bar(positions - width / 2, clean, width, label="Clean")
    axes[0].bar(positions + width / 2, adversarial, width, label="PGD-10")
    axes[0].set_xticks(positions, ["Standard", "Adversarial"])
    axes[0].set(title="Clean and adversarial accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0].legend()

    for data, label in ((standard, "Standard"), (robust, "Adversarial")):
        pairs = sorted((float(eps) * 255, acc) for eps, acc in data["epsilon_analysis"].items())
        if pairs:
            epsilons, accuracies = zip(*pairs)
            axes[1].plot(epsilons, accuracies, "o-", label=label)
    axes[1].set(
        title="Robustness across attack strengths",
        xlabel="Pixel budget (ε × 255)",
        ylabel="Accuracy (%)",
    )
    axes[1].legend()
    success_rates = [stats["attack_success_rate"] for stats in (standard_stats, robust_stats)]
    axes[2].bar(["Standard", "Adversarial"], success_rates, color=["#4c78a8", "#f58518"])
    axes[2].set(title="PGD-10 attack success rate", ylabel="Success rate (%)", ylim=(0, 100))
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(context)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = [
        ("Clean accuracy", "clean_accuracy"),
        ("Adversarial accuracy", "adv_accuracy"),
        ("Attack success rate", "attack_success_rate"),
    ]
    lines = [
        f"# {context}",
        "",
        "| Metric | Standard | Adversarial | Change (percentage points) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        before, after = standard_stats[key], robust_stats[key]
        lines.append(f"| {label} | {before:.2f}% | {after:.2f}% | {after - before:+.2f} |")
    if historical:
        lines.extend(["", "These input files predate the corrected pixel-space attack handling."])
    lines.extend(["", f"Sources: `{standard_path}`, `{robust_path}`.", ""])
    (output / "summary.md").write_text("\n".join(lines))
    print(f"Comparison saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", default="output/analysis_standard.json")
    parser.add_argument("--robust", default="output/analysis_robust.json")
    parser.add_argument("--output-dir", default="output/comparison")
    args = parser.parse_args()
    compare_results(args.standard, args.robust, args.output_dir)
