import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


DATA_DIR = "/content/processed_pill_cls"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
OUTPUT_DIR = "./outputs"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_file_list(data_dir):
    data_dir = Path(data_dir)
    class_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name)
    )

    samples = []
    class_names = []
    class_to_idx = {}

    for idx, class_dir in enumerate(class_dirs):
        original_class_id = int(class_dir.name)
        class_names.append(str(original_class_id))
        class_to_idx[original_class_id] = idx

    for class_dir in class_dirs:
        original_class_id = int(class_dir.name)
        mapped_label = class_to_idx[original_class_id]
        for img_path in class_dir.glob("*.jpg"):
            samples.append((str(img_path), mapped_label))

    return samples, class_names


class PillDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tfms, val_tfms


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Validation", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


def main():
    seed_everything(SEED)
    ensure_dir(OUTPUT_DIR)

    device = get_device()
    print("Using device:", device)

    samples, class_names = build_file_list(DATA_DIR)
    print("Total cropped samples:", len(samples))
    print("Total classes:", len(class_names))

    labels = [label for _, label in samples]

    label_counts = Counter(labels)
    valid_labels = sorted([label for label, count in label_counts.items() if count >= 2])

    samples = [(path, label) for path, label in samples if label in set(valid_labels)]

    # Remap labels to contiguous range: 0..num_classes-1
    old_to_new = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    samples = [(path, old_to_new[label]) for path, label in samples]

    labels = [label for _, label in samples]

    print("Samples after filtering rare classes:", len(samples))
    print("Usable classes:", len(set(labels)))
    print("Min label after remap:", min(labels))
    print("Max label after remap:", max(labels))

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    train_tfms, val_tfms = build_transforms(IMAGE_SIZE)
    train_dataset = PillDataset(train_samples, transform=train_tfms)
    val_dataset = PillDataset(val_samples, transform=val_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    num_classes = len(set(labels))
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes
    ).to(device)

    train_label_counts = Counter([label for _, label in train_samples])
    class_weights = []

    for class_id in range(num_classes):
        count = train_label_counts.get(class_id, 1)
        class_weights.append(1.0 / count)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = 0.0
    best_labels, best_preds = None, None

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, val_labels, val_preds = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Macro F1: {train_f1:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Val   Acc: {val_acc:.4f} | "
            f"Val   Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_labels, best_preds = val_labels, val_preds
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"Saved best model with Val Macro F1 = {best_f1:.4f}")

    print("\nBest Val Macro F1:", round(best_f1, 4))

    if best_labels is not None and best_preds is not None:
        print("\nClassification report on best validation set:")
        print(classification_report(best_labels, best_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
