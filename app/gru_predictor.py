from collections import deque
from typing import Optional

from app.gru_model import GRUModelLoader


class GRUPredictor:
    """
    Maintains a rolling 30-frame feature buffer
    and predicts driver state using the trained GRU model.
    """

    def __init__(self, model_path="models/driveguard_gru.pth"):
        self.model = GRUModelLoader(model_path)
        self.sequence_length = self.model.sequence_length
        self.feature_buffer = deque(maxlen=self.sequence_length)

    def update(
        self,
        ear_left: float,
        ear_right: float,
        ear_avg: float,
        mar: float,
    ) -> Optional[dict]:
        """
        Add one frame of facial features.

        Returns None until 30 frames are available.
        """

        features = [
            float(ear_left),
            float(ear_right),
            float(ear_avg),
            float(mar),
        ]

        self.feature_buffer.append(features)

        # Wait until enough frames are collected
        if len(self.feature_buffer) < self.sequence_length:
            return None

        class_id, class_name, confidence = self.model.predict(
            list(self.feature_buffer)
        )

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "buffer_size": len(self.feature_buffer),
        }

    def reset(self):
        """Clear the rolling frame buffer."""
        self.feature_buffer.clear()