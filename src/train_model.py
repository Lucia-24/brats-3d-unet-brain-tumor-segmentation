from pathlib import Path
import random
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from .augment import augment_patch
except ImportError:
    from augment import augment_patch

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
VAL_FRACTION = 0.2
BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
USE_AUGMENTATION = True
PRINT_EVERY = 10

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# -------------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = PROJECT_ROOT / "patches_binary"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_unet3d_binary.pt"
METRICS_CSV_PATH = MODEL_DIR / "epoch_metrics.csv"

# -----------------------------
# Dataset split
# -----------------------------
def split_ids(patient_ids, val_fraction=0.2, seed=42):
    patient_ids = list(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_val = int(n_total * val_fraction)

    val_ids = patient_ids[:n_val]
    train_ids = patient_ids[n_val:]

    return train_ids, val_ids

# -----------------------------
# Dataset
# -----------------------------
class BraTSPatchDataset(Dataset):
    def __init__(self, patient_ids, patch_dir: Path, augment: bool = False):
        self.patient_ids = list(patient_ids)
        self.patch_dir = patch_dir
        self.augment = augment

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        x_path = self.patch_dir / f"{patient_id}_X_patch.npy"
        y_path = self.patch_dir / f"{patient_id}_y_patch.npy"

        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)

        if self.augment:
            X, y = augment_patch(X, y)

        y = np.expand_dims(y, axis=0)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        return X_tensor, y_tensor

# -----------------------------
# 3D U-Net building blocks
# -----------------------------
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)

        x1 = nn.functional.pad(
            x1,
            [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
                diff_z // 2, diff_z - diff_z // 2,
            ]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.inc = DoubleConv3D(in_channels, 32)
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)

        self.up1 = Up3D(256, 128)
        self.up2 = Up3D(128, 64)
        self.up3 = Up3D(64, 32)

        self.outc = OutConv3D(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits

# -----------------------------
# Loss and metrics
# ------------------------------
def dice_score_from_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs * targets_flat).sum(dim=1)
        union = probs.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return bce_loss + dice_loss

# ------------------------------
# CSV logging helper
# -----------------------------
def initialize_metrics_csv(csv_path: Path) -> None:
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_dice",
            "val_loss",
            "val_dice",
            "train_time_sec",
            "val_time_sec",
            "best_val_dice_so_far"
        ])

def append_metrics_row(
    csv_path: Path,
    epoch: int,
    train_loss: float,
    train_dice: float,
    val_loss: float,
    val_dice: float,
    train_time: float,
    val_time: float,
    best_val_dice: float,
) -> None:
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_dice:.6f}",
            f"{val_loss:.6f}",
            f"{val_dice:.6f}",
            f"{train_time:.2f}",
            f"{val_time:.2f}",
            f"{best_val_dice:.6f}",
        ])

# -----------------------------
# Training / validation loops
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    epoch_start = time.time()

    for batch_idx, (X, y) in enumerate(loader):
        batch_start = time.time()

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_dice = dice_score_from_logits(logits, y).item()
        running_loss += loss.item()
        running_dice += batch_dice

        if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == len(loader):
            print(
                f"[Epoch {epoch_idx + 1}] "
                f"Train batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Dice: {batch_dice:.4f} | "
                f"Batch time: {time.time() - batch_start:.2f}s",
                flush=True
            )

    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_time = time.time() - epoch_start

    return epoch_loss, epoch_dice, epoch_time

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch_idx):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    epoch_start = time.time()

    for batch_idx, (X, y) in enumerate(loader):
        batch_start = time.time()

        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)
        batch_dice = dice_score_from_logits(logits, y).item()

        running_loss += loss.item()
        running_dice += batch_dice

        if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == len(loader):
            print(
                f"[Epoch {epoch_idx + 1}] "
                f"Val batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Dice: {batch_dice:.4f} | "
                f"Batch time: {time.time() - batch_start:.2f}s",
                flush=True
            )

    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_time = time.time() - epoch_start

    return epoch_loss, epoch_dice, epoch_time

def main():
    set_seed(SEED)

    patient_ids_path = PATCH_DIR / "patient_ids.npy"
    if not patient_ids_path.exists():
        raise FileNotFoundError(f"Could not find {patient_ids_path}")

    patient_ids = np.load(patient_ids_path, allow_pickle=True)
    print("Total patient patches found:", len(patient_ids), flush=True)

    train_ids, val_ids = split_ids(patient_ids, val_fraction=VAL_FRACTION, seed=SEED)

    print("Train patients:", len(train_ids), flush=True)
    print("Val patients:", len(val_ids), flush=True)

    train_dataset = BraTSPatchDataset(
        patient_ids=train_ids,
        patch_dir=PATCH_DIR,
        augment=USE_AUGMENTATION,
    )

    val_dataset = BraTSPatchDataset(
        patient_ids=val_ids,
        patch_dir=PATCH_DIR,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("Train batches:", len(train_loader), flush=True)
    print("Val batches:", len(val_loader), flush=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device, flush=True)

    model = UNet3D(in_channels=4, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    initialize_metrics_csv(METRICS_CSV_PATH)
    print(f"Metrics CSV initialized at: {METRICS_CSV_PATH}", flush=True)

    best_val_dice = -1.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_dice, train_time = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_loss, val_dice, val_time = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model to: {BEST_MODEL_PATH}", flush=True)

        append_metrics_row(
            csv_path=METRICS_CSV_PATH,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_dice=train_dice,
            val_loss=val_loss,
            val_dice=val_dice,
            train_time=train_time,
            val_time=val_time,
            best_val_dice=best_val_dice,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"Train Time: {train_time:.2f}s | Val Time: {val_time:.2f}s",
            flush=True
        )

    print("Training complete.", flush=True)
    print("Best validation Dice:", best_val_dice, flush=True)
    print(f"Best model saved at: {BEST_MODEL_PATH}", flush=True)
    print(f"Metrics CSV saved at: {METRICS_CSV_PATH}", flush=True)

if __name__ == "__main__":
    main()