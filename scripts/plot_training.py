"""Plot Training Curves.

Visualizes training progress and convergence behavior.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(log_file, output_path="./output/plots/training_curves.png"):
    """Plot training curves from log file.

    Args:
        log_file: Path to JSON log file with training metrics
        output_path: Path to save the plot
    """
    # Load training log
    with open(log_file, "r") as f:
        log_data = json.load(f)

    epochs = log_data["epochs"]
    train_loss = log_data["train_loss"]
    train_acc = log_data["train_acc"]
    test_clean_acc = log_data["test_clean_acc"]
    test_robust_acc = log_data.get("test_robust_acc", [])

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(epochs, train_loss, "b-", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss Over Time", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Accuracy
    ax2.plot(epochs, train_acc, "g-", linewidth=2, label="Train accuracy", marker="o", markersize=4)
    ax2.plot(
        epochs,
        test_clean_acc,
        "b-",
        linewidth=2,
        label="Test Acc (clean)",
        marker="s",
        markersize=4,
    )
    if test_robust_acc:
        robust_epochs = log_data.get("robust_epochs", epochs[: len(test_robust_acc)])
        ax2.plot(
            robust_epochs,
            test_robust_acc,
            "r-",
            linewidth=2,
            label="Test Acc (robust)",
            marker="^",
            markersize=4,
        )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Over Time", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 100])

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to: {output_path}")

    return fig


def plot_accuracy_tradeoff(
    standard_log, robust_log, output_path="./output/plots/accuracy_tradeoff.png"
):
    """Plot clean vs robust accuracy trade-off.

    Args:
        standard_log: Path to standard training log
        robust_log: Path to robust training log
        output_path: Path to save the plot
    """
    # Load logs
    with open(standard_log, "r") as f:
        std_data = json.load(f)
    with open(robust_log, "r") as f:
        rob_data = json.load(f)

    for log in (std_data, rob_data):
        if not log.get("test_robust_acc"):
            raise ValueError("Both training logs must contain measured test_robust_acc values")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get final accuracies
    std_clean = std_data["test_clean_acc"][-1]
    std_robust = std_data["test_robust_acc"][-1]

    rob_clean = rob_data["test_clean_acc"][-1]
    rob_robust = rob_data["test_robust_acc"][-1]

    # Plot points
    ax.scatter(
        [std_clean],
        [std_robust],
        s=200,
        c="blue",
        marker="o",
        label="Standard Training",
        zorder=3,
        edgecolors="black",
        linewidth=2,
    )
    ax.scatter(
        [rob_clean],
        [rob_robust],
        s=200,
        c="red",
        marker="s",
        label="Adversarial Training",
        zorder=3,
        edgecolors="black",
        linewidth=2,
    )

    # Add annotations
    ax.annotate(
        f"({std_clean:.1f}%, {std_robust:.1f}%)",
        xy=(std_clean, std_robust),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.7),
    )

    ax.annotate(
        f"({rob_clean:.1f}%, {rob_robust:.1f}%)",
        xy=(rob_clean, rob_robust),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", alpha=0.7),
    )

    # Diagonal line (equal accuracy)
    lims = [0, 100]
    ax.plot(lims, lims, "k--", alpha=0.3, zorder=1, label="Clean = Robust")

    ax.set_xlabel("Clean Accuracy (%)", fontsize=12)
    ax.set_ylabel("Robust Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy-Robustness Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Trade-off plot saved to: {output_path}")

    return fig


def main(args):
    if args.log_file:
        # Plot single training curve
        plot_training_curves(args.log_file, args.output)

    if args.standard_log and args.robust_log:
        # Plot trade-off comparison
        plot_accuracy_tradeoff(
            args.standard_log,
            args.robust_log,
            args.output.replace("training_curves", "accuracy_tradeoff"),
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--log-file", type=str, default="", help="Path to training log JSON file")
    parser.add_argument(
        "--standard-log",
        type=str,
        default="",
        help="Path to standard training log (for trade-off plot)",
    )
    parser.add_argument(
        "--robust-log",
        type=str,
        default="",
        help="Path to robust training log (for trade-off plot)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/plots/training_curves.png",
        help="Output path for plot",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot")

    args = parser.parse_args()

    if not args.log_file and not (args.standard_log and args.robust_log):
        print("Error: Provide either --log-file or both --standard-log and --robust-log")
        exit(1)

    main(args)
