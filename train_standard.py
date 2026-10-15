"""Standard Training with Empirical Risk Minimization.

This script implements baseline training using standard ERM:
    min_θ E_{(x,y)~D} [L(θ, x, y)]

This serves as a comparison baseline for the robust training approach.
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


def train_standard(model, train_loader, optimizer, criterion, device, epoch):
    """Standard training loop (ERM).

    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number

    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Standard]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main(args):
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

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Standard Training (Empirical Risk Minimization)")
    print("=" * 60)

    best_clean_acc = -1.0
    history = {key: [] for key in ("epochs", "train_loss", "train_acc", "test_clean_acc")}

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_standard(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Evaluate on clean data
        clean_acc, clean_loss = evaluate_clean(model, test_loader, device)

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {clean_loss:.4f}, Test Acc: {clean_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_clean_acc"].append(clean_acc)

        # Update learning rate
        scheduler.step()

        # Save best model
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                clean_acc,
                os.path.join(args.output_dir, "models", "standard_best.pth"),
            )

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                clean_acc,
                os.path.join(args.output_dir, "models", f"standard_epoch_{epoch}.pth"),
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
    pgd_attack = PGDAttack(
        model, epsilon=args.eval_epsilon, alpha=args.eval_epsilon / 4, num_iter=10, device=device
    )
    robust_acc, _ = evaluate_adversarial(model, test_loader, pgd_attack, device)
    print(f"Robust Accuracy (PGD-10): {robust_acc:.2f}%")
    history["robust_epochs"] = [args.epochs]
    history["test_robust_acc"] = [robust_acc]

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        clean_acc,
        os.path.join(args.output_dir, "models", "standard_final.pth"),
    )

    save_json(history, os.path.join(args.output_dir, "standard_history.json"))

    print("\nTraining completed!")
    print(f"Best Clean Accuracy: {best_clean_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard Training for CIFAR-10")

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
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
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

    # Evaluation parameters
    parser.add_argument(
        "--eval-epsilon", type=float, default=4 / 255, help="Epsilon for adversarial evaluation"
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
