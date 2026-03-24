import os
import random
from collections import Counter

import numpy as np
import pandas as pd
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


CSV_PATH = "/content/processed_pika/pika_metadata.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42
PILL_MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
PRES_MODEL_NAME = "resnet18.a1_in1k"
OUTPUT_DIR = "./outputs_pika"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class PIKADataset(Dataset):
    def __init__(self, df, pill_transform=None, pres_transform=None):
        self.df = df.reset_index(drop=True)
        self.pill_transform = pill_transform
        self.pres_transform = pres_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pill_img = Image.open(row["pill_crop_path"]).convert("RGB")
        pres_img = Image.open(row["prescription_image_path"]).convert("RGB")
        label = int(row["mapped_label"])

        if self.pill_transform:
            pill_img = self.pill_transform(pill_img)
        if self.pres_transform:
            pres_img = self.pres_transform(pres_img)

        return pill_img, pres_img, label


def build_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


class DualEncoderPIKA(nn.Module):
    def __init__(self, num_classes, pill_model_name, pres_model_name):
        super().__init__()

        self.pill_encoder = timm.create_model(
            pill_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        self.pres_encoder = timm.create_model(
            pres_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        pill_dim = self.pill_encoder.num_features
        pres_dim = self.pres_encoder.num_features

        self.fusion = nn.Sequential(
            nn.Linear(pill_dim + pres_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, pill_img, pres_img):
        pill_feat = self.pill_encoder(pill_img)
        pres_feat = self.pres_encoder(pres_img)

        fused = torch.cat([pill_feat, pres_feat], dim=1)
        fused = self.fusion(fused)
        out = self.classifier(fused)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for pill_imgs, pres_imgs, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pill_imgs, pres_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pill_imgs.size(0)
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
    for pill_imgs, pres_imgs, labels in pbar:
        pill_imgs = pill_imgs.to(device)
        pres_imgs = pres_imgs.to(device)
        labels = labels.to(device)

        outputs = model(pill_imgs, pres_imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * pill_imgs.size(0)
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

    df = pd.read_csv(CSV_PATH)
    print("Total metadata rows:", len(df))

    label_counts = Counter(df["pill_label"].tolist())
    valid_labels = sorted([label for label, count in label_counts.items() if count >= 2])

    df = df[df["pill_label"].isin(valid_labels)].copy()
    old_to_new = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    df["mapped_label"] = df["pill_label"].map(old_to_new)

    print("Rows after filtering rare classes:", len(df))
    print("Usable classes:", df["mapped_label"].nunique())
    print("Min mapped label:", int(df["mapped_label"].min()))
    print("Max mapped label:", int(df["mapped_label"].max()))

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["mapped_label"]
    )

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))

    train_tfms, val_tfms = build_transforms(IMAGE_SIZE)

    train_dataset = PIKADataset(train_df, pill_transform=train_tfms, pres_transform=train_tfms)
    val_dataset = PIKADataset(val_df, pill_transform=val_tfms, pres_transform=val_tfms)

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

    num_classes = df["mapped_label"].nunique()
    model = DualEncoderPIKA(
        num_classes=num_classes,
        pill_model_name=PILL_MODEL_NAME,
        pres_model_name=PRES_MODEL_NAME
    ).to(device)

    train_label_counts = Counter(train_df["mapped_label"].tolist())
    class_weights = []
    for class_id in range(num_classes):
        count = train_label_counts.get(class_id, 1)
        class_weights.append(1.0 / count)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_labels, best_preds = val_labels, val_preds
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_pika_model.pth"))
            print(f"Saved best model with Val Macro F1 = {best_f1:.4f}")

    print("\nBest Val Macro F1:", round(best_f1, 4))

    if best_labels is not None and best_preds is not None:
        print("\nClassification report on best validation set:")
        print(classification_report(best_labels, best_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
