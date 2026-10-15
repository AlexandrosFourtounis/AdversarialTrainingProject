"""Plot clean images, pixel-space perturbations, and PGD adversarial examples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from attacks import PGDAttack
from model import CNN
from preprocessing import denormalize
from utils import get_cifar10_loaders, get_device

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def visualize_adversarial_examples(model, test_loader, device, num_images=5, epsilon=4 / 255):
    """Plot the first test examples, including both successful and failed attacks."""
    if num_images < 1:
        raise ValueError("num_images must be positive")
    attack = PGDAttack(model, epsilon=epsilon, alpha=epsilon / 4, device=device)
    examples = []
    model.eval()
    for images, labels in test_loader:
        remaining = num_images - len(examples)
        images, labels = images[:remaining].to(device), labels[:remaining].to(device)
        adversarial = attack(images, labels)
        with torch.no_grad():
            clean_predictions = model(images).argmax(1)
            adversarial_predictions = model(adversarial).argmax(1)
        for i in range(len(images)):
            examples.append(
                (
                    denormalize(images[i]).cpu(),
                    denormalize(adversarial[i]).cpu(),
                    labels[i].item(),
                    clean_predictions[i].item(),
                    adversarial_predictions[i].item(),
                )
            )
        if len(examples) >= num_images:
            break
    if not examples:
        raise ValueError("The test loader contains no images")

    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3 * len(examples)), squeeze=False)
    for row, (clean, adversarial, label, clean_prediction, adversarial_prediction) in enumerate(
        examples
    ):
        perturbation = (adversarial - clean).abs()
        perturbation = perturbation / perturbation.max().clamp_min(1e-12)
        images = (clean.clamp(0, 1), perturbation, adversarial.clamp(0, 1))
        titles = (
            f"Clean: {CIFAR10_CLASSES[clean_prediction]}\nTrue: {CIFAR10_CLASSES[label]}",
            f"Absolute perturbation (amplified)\nε = {epsilon * 255:g}/255",
            f"Adversarial: {CIFAR10_CLASSES[adversarial_prediction]}",
        )
        for column, (image, title) in enumerate(zip(images, titles)):
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row, column].set_title(title, fontsize=10)
            axes[row, column].axis("off")
    fig.tight_layout()
    return fig


def main(args):
    torch.manual_seed(args.seed)
    device = get_device()
    model = CNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
    fig = visualize_adversarial_examples(
        model,
        test_loader,
        device,
        num_images=args.num_images,
        epsilon=args.epsilon,
    )
    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=4 / 255)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output/visualizations")
    parser.add_argument("--output-name", default="adversarial_examples.png")
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())
