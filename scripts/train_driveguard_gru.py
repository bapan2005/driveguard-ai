import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

X_PATH = "data/X_temporal.npy"
Y_PATH = "data/y_temporal.npy"

MODEL_PATH = "models/driveguard_gru.pth"

SEQUENCE_LENGTH = 30
INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 4

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

RANDOM_SEED = 42


CLASS_NAMES = [
    "NORMAL",
    "YAWNING",
    "DROWSY",
    "MICROSLEEP",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class TemporalFatigueDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ============================================================
# GRU Model
# ============================================================

class DriveGuardGRU(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_step = output[:, -1, :]

        logits = self.classifier(last_step)

        return logits


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch_x.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            batch_y.detach().cpu().numpy()
        )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return average_loss, accuracy


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            total_loss += (
                loss.item()
                * batch_x.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch_y.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        matrix,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    print("=" * 65)
    print("DriveGuard AI - PyTorch GRU Training")
    print("=" * 65)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load temporal dataset
    # --------------------------------------------------------

    if not os.path.exists(X_PATH):
        raise FileNotFoundError(
            f"Missing: {X_PATH}\n"
            "Run prepare_temporal_dataset.py first."
        )

    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(
            f"Missing: {Y_PATH}\n"
            "Run prepare_temporal_dataset.py first."
        )

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to have 3 dimensions, got {X.shape}"
        )

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, got {X.shape[1]}"
        )

    if X.shape[2] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"got {X.shape[2]}"
        )

    # --------------------------------------------------------
    # Train / validation / test split
    #
    # IMPORTANT:
    # This is a stratified split of temporal sequences.
    # We will later improve this to a strict recording-session
    # split once session IDs are stored in the dataset.
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print("\nDataset split:")
    print(f"  Training:   {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Testing:    {len(X_test)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TemporalFatigueDataset(
        X_train,
        y_train,
    )

    val_dataset = TemporalFatigueDataset(
        X_val,
        y_val,
    )

    test_dataset = TemporalFatigueDataset(
        X_test,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DriveGuardGRU().to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    best_val_accuracy = 0.0

    print("\n" + "=" * 65)
    print("Training")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}% | "
            f"Val F1: {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint = {
                "state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "num_classes": NUM_CLASSES,
                "sequence_length": SEQUENCE_LENGTH,
                "class_names": CLASS_NAMES,
            }

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

            print(
                f"  -> Saved best model: "
                f"{MODEL_PATH}"
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        confusion,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(
        f"\nTest Loss:      {test_loss:.4f}"
    )

    print(
        f"Test Accuracy:  {test_accuracy * 100:.2f}%"
    )

    print(
        f"Precision:      {test_precision:.4f}"
    )

    print(
        f"Recall:         {test_recall:.4f}"
    )

    print(
        f"F1 Score:       {test_f1:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")
    print(
        "Rows = Actual | Columns = Predicted"
    )

    print(
        "              "
        + " ".join(
            f"{name[:6]:>8}"
            for name in CLASS_NAMES
        )
    )

    for index, row in enumerate(confusion):

        print(
            f"{CLASS_NAMES[index]:12}"
            + " ".join(
                f"{value:8d}"
                for value in row
            )
        )

    print("\nModel saved to:")
    print(f"  {MODEL_PATH}")


if __name__ == "__main__":
    main()