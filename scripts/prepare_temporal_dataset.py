import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "data/driver_fatigue_dataset.csv"

OUTPUT_X = "data/X_temporal.npy"
OUTPUT_Y = "data/y_temporal.npy"

SEQUENCE_LENGTH = 30

FEATURE_COLUMNS = [
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
]

LABEL_COLUMN = "label"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DriveGuard AI - Temporal Dataset Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print(f"\nLoaded dataset: {CSV_PATH}")
    print(f"Total rows: {len(df)}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    after = len(df)

    if before != after:
        print(
            f"Removed {before - after} rows containing missing values."
        )

    # --------------------------------------------------------
    # Identify recording blocks
    #
    # A new block starts whenever the label changes.
    #
    # We DO NOT allow sequences to cross a block boundary.
    # --------------------------------------------------------

    df["_block_id"] = (
        df[LABEL_COLUMN]
        .ne(df[LABEL_COLUMN].shift())
        .cumsum()
    )

    number_of_blocks = df["_block_id"].nunique()

    print(f"\nRecording blocks found: {number_of_blocks}")

    # --------------------------------------------------------
    # Create temporal sequences
    # --------------------------------------------------------

    X_sequences = []
    y_sequences = []

    skipped_blocks = 0

    for block_id, block in df.groupby("_block_id"):

        label = int(block[LABEL_COLUMN].iloc[0])

        features = block[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        number_of_frames = len(features)

        number_of_sequences = (
            number_of_frames // SEQUENCE_LENGTH
        )

        if number_of_sequences == 0:
            skipped_blocks += 1
            continue

        print(
            f"Block {block_id:02d} | "
            f"Label: {label} | "
            f"Frames: {number_of_frames} | "
            f"Sequences: {number_of_sequences}"
        )

        # ----------------------------------------------------
        # Non-overlapping 30-frame sequences
        # ----------------------------------------------------

        for sequence_index in range(number_of_sequences):

            start = (
                sequence_index
                * SEQUENCE_LENGTH
            )

            end = (
                start
                + SEQUENCE_LENGTH
            )

            sequence = features[
                start:end
            ]

            # Safety check
            if sequence.shape != (
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS),
            ):
                continue

            X_sequences.append(sequence)
            y_sequences.append(label)

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X = np.asarray(
        X_sequences,
        dtype=np.float32
    )

    y = np.asarray(
        y_sequences,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Validate final shapes
    # --------------------------------------------------------

    expected_feature_shape = (
        SEQUENCE_LENGTH,
        len(FEATURE_COLUMNS),
    )

    if X.ndim != 3:
        raise RuntimeError(
            f"Unexpected X dimensions: {X.shape}"
        )

    if X.shape[1:] != expected_feature_shape:
        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if len(X) != len(y):
        raise RuntimeError(
            "X and y have different numbers of samples."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Temporal dataset created successfully")
    print("=" * 60)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nClass distribution:")

    class_names = {
        0: "NORMAL",
        1: "YAWNING",
        2: "DROWSY",
        3: "MICROSLEEP",
    }

    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique_labels,
        counts
    ):
        print(
            f"  {label} = "
            f"{class_names.get(int(label), 'UNKNOWN'):10s} "
            f": {count}"
        )

    print(
        f"\nSkipped blocks: {skipped_blocks}"
    )

    print(f"\nSaved:")
    print(f"  {OUTPUT_X}")
    print(f"  {OUTPUT_Y}")


if __name__ == "__main__":
    main()