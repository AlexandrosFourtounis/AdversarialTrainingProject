"""Comprehensive Robustness Analysis.

Evaluates models against different attack strengths and provides detailed analysis.
"""

import argparse

import torch
from tqdm import tqdm

from attacks import PGDAttack
from model import CNN
from utils import get_cifar10_loaders, get_device, save_json


def evaluate_multiple_epsilons(
    model, test_loader, device, epsilons=[2 / 255, 4 / 255, 6 / 255, 8 / 255]
):
    """Evaluate model against PGD attacks with different epsilon values.

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to run on
        epsilons: List of epsilon values to test

    Returns:
        Dictionary mapping epsilon to accuracy
    """
    results = {}

    for eps in epsilons:
        print(f"\n{'=' * 60}")
        print(f"Evaluating against PGD-10 with ε = {eps:.4f} ({eps * 255:.1f}/255)")
        print(f"{'=' * 60}")

        # Create attack with this epsilon
        attack = PGDAttack(
            model=model,
            epsilon=eps,
            alpha=eps / 4,  # Step size = epsilon/4
            num_iter=10,
            device=device,
        )

        model.eval()
        correct = 0
        total = 0

        for images, labels in tqdm(test_loader, desc=f"ε={eps:.4f}"):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            images_adv = attack(images, labels, random_start=True)

            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(images_adv)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        results[eps] = accuracy
        print(f"Robust Accuracy (ε={eps:.4f}): {accuracy:.2f}%")

    return results


def evaluate_attack_success_rate(model, test_loader, device, epsilon=4 / 255):
    """Calculate attack success rate - how many clean images are misclassified after attack.

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to run on
        epsilon: Attack strength

    Returns:
        Dictionary with detailed attack statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Attack Success Rate Analysis (ε = {epsilon:.4f})")
    print(f"{'=' * 60}")

    attack = PGDAttack(model=model, epsilon=epsilon, alpha=epsilon / 4, num_iter=10, device=device)

    model.eval()

    total = 0
    clean_correct = 0
    adv_correct = 0
    successful_attacks = 0  # Clean correct but adv wrong

    for images, labels in tqdm(test_loader, desc="Analyzing attacks"):
        images, labels = images.to(device), labels.to(device)

        # Evaluate on clean images
        with torch.no_grad():
            outputs_clean = model(images)
            _, pred_clean = outputs_clean.max(1)

        # Generate adversarial examples
        images_adv = attack(images, labels, random_start=True)

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs_adv = model(images_adv)
            _, pred_adv = outputs_adv.max(1)

        # Statistics
        total += labels.size(0)
        clean_correct += pred_clean.eq(labels).sum().item()
        adv_correct += pred_adv.eq(labels).sum().item()

        # Successful attack = clean correct AND adv wrong
        successful_attacks += (pred_clean.eq(labels) & ~pred_adv.eq(labels)).sum().item()

    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    attack_success = 100.0 * successful_attacks / clean_correct if clean_correct > 0 else 0

    results = {
        "total_samples": total,
        "clean_correct": clean_correct,
        "clean_accuracy": clean_acc,
        "adv_correct": adv_correct,
        "adv_accuracy": adv_acc,
        "successful_attacks": successful_attacks,
        "attack_success_rate": attack_success,
    }

    print(f"\nClean Accuracy: {clean_acc:.2f}% ({clean_correct}/{total})")
    print(f"Adversarial Accuracy: {adv_acc:.2f}% ({adv_correct}/{total})")
    print(f"Successful Attacks: {successful_attacks}/{clean_correct} images")
    print(f"Attack Success Rate: {attack_success:.2f}%")
    print(f"  (Out of {clean_correct} correctly classified clean images,")
    print(f"   {successful_attacks} were fooled by the attack)")

    return results


def main(args):
    # Set device
    device = get_device()
    torch.manual_seed(args.seed)
    print(f"Using device: {device}\n")

    # Load data
    print("Loading CIFAR-10 dataset...")
    _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir, num_workers=args.num_workers
    )

    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        from torch.utils.data import DataLoader, Subset

        subset = Subset(test_loader.dataset, range(min(args.max_samples, len(test_loader.dataset))))
        test_loader = DataLoader(subset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize model
    model = CNN().to(device)

    # Load checkpoint
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("\n" + "=" * 60)
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print("=" * 60)

    # 1. Evaluate against different epsilon values
    print("\n[1/2] Testing Different Attack Strengths...")
    epsilon_results = evaluate_multiple_epsilons(model, test_loader, device, epsilons=args.epsilons)

    # 2. Calculate attack success rate
    print("\n[2/2] Calculating Attack Success Rate...")
    attack_stats = evaluate_attack_success_rate(model, test_loader, device, epsilon=4 / 255)

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print("\nRobustness vs Attack Strength:")
    for eps, acc in epsilon_results.items():
        print(f"  ε = {eps:.4f} ({eps * 255:4.1f}/255): {acc:5.2f}%")

    print("\nAttack Effectiveness (ε = 4/255):")
    print(f"  Attack Success Rate: {attack_stats['attack_success_rate']:.2f}%")
    print(
        f"  Clean → Adversarial: {attack_stats['clean_accuracy']:.2f}% → {attack_stats['adv_accuracy']:.2f}%"
    )

    # Save results
    if args.save_results:
        results = {
            "evaluation": {
                "attack": "PGD-10",
                "attack_space": "pixel",
                "epsilon": 4 / 255,
                "step_size": "epsilon/4",
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
            },
            "epsilon_analysis": {str(k): v for k, v in epsilon_results.items()},
            "attack_success_analysis": attack_stats,
        }
        save_json(results, args.save_results)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive robustness analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[2 / 255, 4 / 255, 6 / 255, 8 / 255],
        help="Epsilon values to test",
    )
    parser.add_argument(
        "--save-results", type=str, default="", help="Path to save analysis results (JSON)"
    )

    parser.add_argument("--data-dir", default="./data", help="CIFAR-10 dataset directory")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loading workers")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation samples for a quick smoke run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PGD starts")
    args = parser.parse_args()
    main(args)
