"""Shared CIFAR-10 normalization for training, attacks, and visualization."""

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def normalize(images):
    """Normalize CHW or NCHW pixel tensors without changing device or dtype."""
    mean = images.new_tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(3, 1, 1)
    return (images - mean) / std


def denormalize(images):
    """Convert normalized CHW or NCHW tensors back to pixel values."""
    mean = images.new_tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(3, 1, 1)
    return images * std + mean
