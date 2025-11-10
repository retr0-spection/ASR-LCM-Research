import os
import pandas as pd

csv_path = "/datasets/onailana/test_codes/test_code_data.csv"
codes_dir = "/datasets/onailana/test_codes"

# Load existing CSV
df = pd.read_csv(csv_path)

# If filepath column missing, add it
if "filepath" not in df.columns:
    df["filepath"] = df["filename"].apply(
        lambda f: os.path.join(codes_dir, f.replace(".wav", ".pt"))
    )

# Save back
df.to_csv(csv_path, index=False)
print(f"✅ Patched CSV saved at {csv_path}")

