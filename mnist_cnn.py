import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=download
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_sum += loss.item() * inputs.size(0)
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simple CNN on MNIST using PyTorch"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="/workspace/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--save-path", type=str, default="/workspace/artifacts/mnist_cnn_best.pt"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    try:
        train_loader, test_loader = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=2,
            download=not args.no_download,
        )
    except Exception as exc:
        print(f"Failed to prepare MNIST: {exc}")
        print(
            "If running offline, place MNIST data under the data dir and use --no-download."
        )
        raise

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(
                f"Saved new best model to {args.save_path} (val_acc={best_acc:.4f})"
            )

    print(f"Best val_acc: {best_acc:.4f}")

    model.eval()
    sample_inputs, sample_targets = next(iter(test_loader))
    sample_inputs = sample_inputs[:10].to(device)
    sample_targets = sample_targets[:10].to(device)
    with torch.no_grad():
        outputs = model(sample_inputs)
        preds = outputs.argmax(dim=1)
    print("Sample ground truth:", sample_targets.tolist())
    print("Sample predictions:", preds.tolist())


if __name__ == "__main__":
    main()