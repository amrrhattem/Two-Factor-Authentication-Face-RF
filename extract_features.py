# ============================================================
# FILE 5: extract_features.py
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
from config import ALL_FEATURES_PATH

# --- Configure these paths via environment or edit here ---
DATASET_PATH = os.environ.get('RFFI_DATASET_PATH', './FE-Dataset')
OUTPUT_PATH = os.path.dirname(ALL_FEATURES_PATH)

os.makedirs(OUTPUT_PATH, exist_ok=True)

FEATURE_COLS = [
    'CFO', 'short_freq', 'long_freq', 'frac_dimension_1', 'frac_dimension_2',
    'iqi_1', 'iqi_2', 'mag_error_mean_1', 'mag_error_var_1',
    'mag_error_mean_2', 'mag_error_var_2', 'phase_error_mean_1',
    'phase_error_var_1', 'phase_error_mean_2', 'phase_error_var_2'
]

files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]
device_features = {}
failed = 0

for file in files:
    filepath = os.path.join(DATASET_PATH, file)
    try:
        df = pd.read_csv(filepath)

        mac_cols = [col for col in df.columns if 'MAC' in col.upper()]
        if mac_cols:
            mac = str(df[mac_cols[0]].iloc[0]).replace(':', '_')
        else:
            mac = file.replace('_pre.csv', '').replace('.csv', '')
            print(f"Using filename as MAC: {mac}")

        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        if not available_cols:
            print(f"⚠️  {file}: No feature columns found, skipping...")
            failed += 1
            continue

        avg_features = df[available_cols].values.mean(axis=0)
        device_features[mac] = avg_features

        # Save individual device file
        device_out = os.path.join(OUTPUT_PATH, f"{mac}_features.pkl")
        with open(device_out, 'wb') as f:
            pickle.dump(avg_features, f)

        print(f"✅ {mac}: {len(avg_features)} features from {len(df)} rows")

    except Exception as e:
        print(f"❌ Error in {file}: {e}")
        failed += 1

print(f"\n✅ Total devices: {len(device_features)}")
print(f"❌ Failed: {failed}")

if device_features:
    with open(ALL_FEATURES_PATH, 'wb') as f:
        pickle.dump(device_features, f)
    print(f"✅ Saved to {ALL_FEATURES_PATH}")
else:
    print("❌ No features extracted — check RFFI_DATASET_PATH")
