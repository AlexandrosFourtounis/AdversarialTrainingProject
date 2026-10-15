"""Model Architecture for CIFAR-10 Classification."""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Convolutional Neural Network for CIFAR-10.

    A simple but effective architecture with:
    - 2 convolutional blocks with batch normalization
    - 2 fully connected layers
    - Dropout for regularization
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Second convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # First block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Second block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
