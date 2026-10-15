"""Adversarial fine-tuning with a configurable mixture of clean and PGD losses.

The adversarial loss weight is controlled by --adv-ratio; 1.0 uses only
adversarial examples, and 0.75 gives 75% adversarial / 25% clean loss.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from attacks import PGDAttack
from model import CNN
from utils import (
    evaluate_adversarial,
    evaluate_clean,
    get_cifar10_loaders,
    get_device,
    save_checkpoint,
    save_json,
)


def train_robust(model, train_loader, optimizer, criterion, attack, device, epoch, adv_ratio=0.5):
    """Train one epoch using the requested adversarial loss weight.

    Accuracy is measured on adversarial examples when adv_ratio > 0;
    otherwise it is measured on clean examples. Loss is the weighted mean.
    """
    if not 0 <= adv_ratio <= 1:
        raise ValueError("adv_ratio must be between 0 and 1")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Robust]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if adv_ratio > 0:
            images_adv = attack(images, labels, random_start=True)
            outputs = model(images_adv)
            loss = adv_ratio * criterion(outputs, labels)
        if adv_ratio < 1:
            clean_outputs = model(images)
            clean_loss = (1 - adv_ratio) * criterion(clean_outputs, labels)
            if adv_ratio == 0:
                outputs, loss = clean_outputs, clean_loss
            else:
                loss = loss + clean_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        pbar.set_postfix(loss=running_loss / (batch_idx + 1), acc=100 * correct / total)
    return running_loss / len(train_loader), 100 * correct / total


def main(args):
    if not 0 <= args.adv_ratio <= 1:
        raise ValueError("--adv-ratio must be between 0 and 1")
    if args.eval_freq < 1:
        raise ValueError("--eval-freq must be positive")
    if args.pretrain_path and not os.path.isfile(args.pretrain_path):
        raise FileNotFoundError(f"Pre-training checkpoint not found: {args.pretrain_path}")
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir
    )

    # Initialize model
    print("Initializing model...")
    model = CNN(num_classes=10, dropout_rate=args.dropout).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.pretrain_path:
        print(f"Loading pre-trained weights from: {args.pretrain_path}")
        checkpoint = torch.load(args.pretrain_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        pretrain_clean_acc, _ = evaluate_clean(model, test_loader, device)
        print(f"Pre-trained clean accuracy: {pretrain_clean_acc:.2f}%")

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )

    # Initialize PGD attack for training
    print("\nPGD Attack Configuration:")
    print(f"  Epsilon: {args.train_epsilon:.4f} ({args.train_epsilon * 255:.1f}/255)")
    print(f"  Alpha: {args.train_alpha:.4f}")
    print(f"  Iterations: {args.train_steps}")

    train_attack = PGDAttack(
        model=model,
        epsilon=args.train_epsilon,
        alpha=args.train_alpha,
        num_iter=args.train_steps,
        device=device,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Robust Training (Min-Max Saddle Point Optimization)")
    print("=" * 60)

    best_robust_acc = -1.0
    history = {
        key: []
        for key in (
            "epochs",
            "train_loss",
            "train_acc",
            "test_clean_acc",
            "robust_epochs",
            "test_robust_acc",
        )
    }

    for epoch in range(1, args.epochs + 1):
        # Train with adversarial examples
        train_loss, train_acc = train_robust(
            model, train_loader, optimizer, criterion, train_attack, device, epoch, args.adv_ratio
        )

        # Evaluate on clean data
        clean_acc, clean_loss = evaluate_clean(model, test_loader, device)

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss (weighted): {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss (clean): {clean_loss:.4f}, Test Acc (clean): {clean_acc:.2f}%")

        # Periodic robust evaluation
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            print("  Running robust evaluation...")
            eval_attack = PGDAttack(
                model=model,
                epsilon=args.eval_epsilon,
                alpha=args.eval_epsilon / 4,
                num_iter=10,
                device=device,
            )
            robust_acc, robust_loss = evaluate_adversarial(model, test_loader, eval_attack, device)
            print(f"  Test Loss (PGD-10): {robust_loss:.4f}, Test Acc (PGD-10): {robust_acc:.2f}%")

            history["robust_epochs"].append(epoch)
            history["test_robust_acc"].append(robust_acc)

            # Save best robust model
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    robust_acc,
                    os.path.join(args.output_dir, "models", "robust_best.pth"),
                )

        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_clean_acc"].append(clean_acc)

        # Update learning rate
        scheduler.step()

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                clean_acc,
                os.path.join(args.output_dir, "models", f"robust_epoch_{epoch}.pth"),
            )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Clean accuracy
    clean_acc, _ = evaluate_clean(model, test_loader, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    # Adversarial accuracy (PGD-10)
    print("\nEvaluating against PGD-10 attack...")
    final_attack = PGDAttack(
        model=model,
        epsilon=args.eval_epsilon,
        alpha=args.eval_epsilon / 4,
        num_iter=10,
        device=device,
    )
    robust_acc, _ = evaluate_adversarial(model, test_loader, final_attack, device)
    print(f"Robust Accuracy (PGD-10): {robust_acc:.2f}%")

    # Adversarial accuracy (PGD-20)
    print("\nEvaluating against PGD-20 attack (stronger)...")
    strong_attack = PGDAttack(
        model=model,
        epsilon=args.eval_epsilon,
        alpha=args.eval_epsilon / 4,
        num_iter=20,
        device=device,
    )
    robust_acc_20, _ = evaluate_adversarial(model, test_loader, strong_attack, device)
    print(f"Robust Accuracy (PGD-20): {robust_acc_20:.2f}%")

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        robust_acc,
        os.path.join(args.output_dir, "models", "robust_final.pth"),
    )

    save_json(history, os.path.join(args.output_dir, "robust_history.json"))

    print("\nTraining completed!")
    print(f"Best Robust Accuracy (PGD-10): {best_robust_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Training for CIFAR-10")

    # Data parameters
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Directory for CIFAR-10 dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Directory for saving outputs"
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate (use 0.01 for pre-training, 0.1 from scratch)",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay (L2 regularization)"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Learning rate schedule
    parser.add_argument(
        "--lr-milestones",
        type=int,
        nargs="+",
        default=[75, 90],
        help="Epochs to decrease learning rate",
    )
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Learning rate decay factor")

    # Adversarial training parameters (inner maximization)
    parser.add_argument(
        "--train-epsilon",
        type=float,
        default=4 / 255,
        help="Perturbation bound for training (L-inf norm)",
    )
    parser.add_argument(
        "--train-alpha", type=float, default=2 / 255, help="Step size for PGD during training"
    )
    parser.add_argument(
        "--train-steps", type=int, default=10, help="Number of PGD steps during training"
    )
    parser.add_argument(
        "--adv-ratio",
        type=float,
        default=0.5,
        help="Adversarial loss weight (0.5 = equal weights, 0.75 = 75-25, 1.0 = pure adversarial)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-epsilon", type=float, default=4 / 255, help="Epsilon for adversarial evaluation"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10, help="Evaluate robust accuracy every N epochs"
    )

    # Pre-training parameters
    parser.add_argument(
        "--pretrain-path",
        type=str,
        default=None,
        help="Path to pre-trained standard model (e.g., output/models/standard_final.pth)",
    )

    # Other parameters
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of workers for data loading"
    )
    parser.add_argument("--save-freq", type=int, default=25, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    main(args)
