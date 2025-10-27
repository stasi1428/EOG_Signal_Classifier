#!/usr/bin/env python3
"""
Export EOG .txt files into a unified CSV.

Each row in the output CSV corresponds to one sample point in one trial:
filename | label | channel | trial_id | timestep | value
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Map prefixes to canonical labels
LABELS_MAP = {
    "yukari": "Up",
    "asagi": "Down",
    "sag": "Right",
    "sol": "Left",
    "kirp": "Blink"
}

def load_and_export(data_dir, out_path):
    data_dir = Path(data_dir)
    rows = []

    for file in data_dir.glob("*.txt"):
        name = file.stem.lower()  # e.g., "asagi1h"
        # Identify class
        label = None
        for key, val in LABELS_MAP.items():
            if name.startswith(key):
                label = val
                break
        if label is None:
            # Skip files not in our defined classes
            continue

        # Identify channel
        channel = "horizontal" if name.endswith("h") else "vertical" if name.endswith("v") else "unknown"
        # Extract trial number
        trial_id = ''.join([c for c in name if c.isdigit()]) or None

        # Load signal
        signal = np.loadtxt(file).astype(float)

        # Flatten into rows
        for i, val in enumerate(signal):
            rows.append({
                "filename": file.name,
                "label": label,
                "channel": channel,
                "trial_id": trial_id,
                "timestep": i,
                "value": val
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Exported {len(df)} rows from {data_dir} to {out_path}")

if __name__ == "__main__":
    # Adjust paths as needed
    data_dir = r"C:\Users\Stasis Mukwenya\Documents\GitHub\EOG\data\eog_dataset"
    out_path = Path(data_dir).parent / "eog_dataset.csv"
    load_and_export(data_dir, out_path)