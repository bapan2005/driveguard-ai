import os
from typing import Tuple

import torch
import torch.nn as nn


class DriveGuardGRU(nn.Module):
    """Lightweight GRU used for DriveGuard fatigue classification."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
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

        return self.classifier(last_step)


class GRUModelLoader:
    """Loads the trained DriveGuard GRU checkpoint."""

    CLASS_NAMES = {
        0: "NORMAL",
        1: "YAWNING",
        2: "DROWSY",
        3: "MICROSLEEP",
    }

    def __init__(
        self,
        model_path: str = "models/driveguard_gru.pth",
    ):
        self.model_path = model_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GRU model not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.input_size = checkpoint.get(
            "input_size",
            4,
        )

        self.hidden_size = checkpoint.get(
            "hidden_size",
            64,
        )

        self.num_layers = checkpoint.get(
            "num_layers",
            2,
        )

        self.num_classes = checkpoint.get(
            "num_classes",
            4,
        )

        self.sequence_length = checkpoint.get(
            "sequence_length",
            30,
        )

        self.model = DriveGuardGRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
        )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    def predict(
        self,
        sequence,
    ) -> Tuple[int, str, float]:

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32,
            device=self.device,
        )

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        class_id = int(
            prediction.item()
        )

        class_name = self.CLASS_NAMES.get(
            class_id,
            "UNKNOWN",
        )

        confidence_value = float(
            confidence.item()
        )

        return (
            class_id,
            class_name,
            confidence_value,
        )