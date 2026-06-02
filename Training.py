
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat


material = input("Enter material name (e.g., soil, road): ").strip().lower()
IMAGE_PATH = input("Enter the .mat hyperspectral file path: ").strip()
GT_INPUT_PATH = input("Enter GT .mat file path (shadow/material labels): ").strip()

NUM_SHADOW = int(input("How many SHADOW points? "))
NUM_OBJECT = int(input("How many NON-SHADOW points? "))


def load_hsi(path):
    data = loadmat(path)
    cube = next(v for v in data.values()
                if isinstance(v, np.ndarray) and v.ndim == 3)

    cube = cube.astype(float)

    if cube.shape[0] < cube.shape[2]:
        cube = cube.transpose(1, 2, 0)

    return cube

img = load_hsi(IMAGE_PATH)
H, W, B = img.shape



gt_data = loadmat(GT_INPUT_PATH)
gt_raw = next(v for v in gt_data.values()
              if isinstance(v, np.ndarray) and v.shape[:2] == (H, W))

gt_raw = gt_raw.astype(np.uint8)


gt_mask = np.zeros((H, W), dtype=np.uint8)
gt_mask[gt_raw == 1] = 1 
gt_mask[gt_raw == 2] = 2 

savemat(f"{material}_train_gt.mat", {"gt_mask": gt_mask})
print("✓ Saved standardized GT:", f"{material}_train_gt.mat")


def minmax(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


rgb_bands = [int(B * 0.2), int(B * 0.5), int(B * 0.8)]
rgb = minmax(img[:, :, rgb_bands])


plt.figure(figsize=(10, 8))
plt.imshow(rgb)
plt.title(f"Click {NUM_SHADOW} SHADOW pixels")
shadow_clicks = plt.ginput(NUM_SHADOW, timeout=-1)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(rgb)
plt.title(f"Click {NUM_OBJECT} NON-SHADOW pixels")
object_clicks = plt.ginput(NUM_OBJECT, timeout=-1)
plt.close()


def collect(clicks, label):
    rows = []
    for (x, y) in clicks:
        xi, yi = int(x), int(y)
        raw = img[yi, xi, :]
        norm = minmax(raw)

        row = {"material": material, "class": label, "x": xi, "y": yi}
        for i in range(B):
            row[f"band_{i}"] = raw[i]
            row[f"norm_band_{i}"] = norm[i]
        rows.append(row)

    return pd.DataFrame(rows)

shadow_df = collect(shadow_clicks, "shadow")
material_df = collect(object_clicks, "material")

shadow_df.to_csv(f"{material}_shadow.csv", index=False)
material_df.to_csv(f"{material}_material.csv", index=False)

print("✓ Sample CSVs saved")



bands = np.arange(B)

for name, df in [("shadow", shadow_df), ("material", material_df)]:
    plt.figure(figsize=(10, 5))
    for i in range(len(df)):
        plt.plot(bands, df.filter(like="norm_band_").iloc[i], alpha=0.3)
    plt.plot(bands, df.filter(like="norm_band_").mean(), color="black", lw=3)
    plt.title(f"{material} {name.upper()} spectrum")
    plt.savefig(f"{material}_{name}_plot.png", dpi=300)
    plt.close()

print("✓ Data collection complete")