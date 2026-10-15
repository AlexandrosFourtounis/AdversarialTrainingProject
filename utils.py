"""Utility functions for data loading, training, and evaluation."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing import CIFAR10_MEAN, CIFAR10_STD


def get_cifar10_loaders(batch_size=128, num_workers=2, data_dir="./data"):
    """Load CIFAR-10 dataset with standard preprocessing.

    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load dataset

    Returns:
        train_loader, test_loader
    """
    # Training transforms (with data augmentation)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # Test transforms (no augmentation)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Create data loaders
    # Note: pin_memory=False for MPS compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader


def evaluate_clean(model, test_loader, device):
    """Evaluate model on clean (natural) test data.

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        accuracy: Clean accuracy
        avg_loss: Average loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


def evaluate_adversarial(model, test_loader, attack, device):
    """Evaluate model on adversarial examples.

    Args:
        model: Neural network model
        test_loader: Test data loader
        attack: Attack object (e.g., PGDAttack)
        device: Device to run evaluation on

    Returns:
        accuracy: Robust accuracy (on adversarial examples)
        avg_loss: Average loss on adversarial examples
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in tqdm(test_loader, desc="Adversarial Eval"):
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples
        images_adv = attack(images, labels)

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(images_adv)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """Save model checkpoint.

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint.

    Args:
        model: Neural network model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load model on

    Returns:
        epoch: Last epoch number
        accuracy: Last accuracy
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    accuracy = checkpoint["accuracy"]

    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Accuracy {accuracy:.2f}%)")
    return epoch, accuracy


def get_device():
    """Select Apple Silicon MPS, CUDA, or CPU consistently across commands."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data, filepath):
    """Save metrics, creating the destination directory when needed."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
