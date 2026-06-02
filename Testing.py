
import numpy as np
import pandas as pd



def SAM_vec(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    An = np.linalg.norm(A, axis=1, keepdims=True)
    Bn = np.linalg.norm(B, axis=1, keepdims=True)
    cosang = (A @ B.T) / (An * Bn.T + 1e-12)
    return np.arccos(np.clip(cosang, -1, 1))


def SID_vec(A, B):
    eps = 1e-12
    A = np.clip(A, eps, None)
    B = np.clip(B, eps, None)
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    KL_AB = np.sum(A[:, None, :] * np.log(A[:, None, :] / B[None, :, :]), axis=2)
    KL_BA = np.sum(B[None, :, :] * np.log(B[None, :, :] / A[:, None, :]), axis=2)
    return KL_AB + KL_BA


def SCC_vec(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    A -= A.mean(axis=1, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)
    return (A @ B.T) / (
        np.sqrt((A**2).sum(axis=1, keepdims=True) * (B**2).sum(axis=1)) + 1e-12
    )


def compute_threshold(intra, inter, higher_is_better=False):
    intra_vals = intra[np.triu_indices_from(intra, k=1)]
    inter_vals = inter.flatten()

    if higher_is_better:
        t = (np.percentile(intra_vals, 5) + np.percentile(inter_vals, 95)) / 2
    else:
        t = (np.percentile(intra_vals, 95) + np.percentile(inter_vals, 5)) / 2

    return float(t)


material = input("Enter material name: ").strip().lower()

shadow_df = pd.read_csv(f"{material}_shadow.csv")
material_df = pd.read_csv(f"{material}_material.csv")

band_cols = [c for c in shadow_df.columns if c.startswith("band_")]

shadow = shadow_df[band_cols].values.astype(float)
material_cls = material_df[band_cols].values.astype(float)

print("\nSamples loaded:")
print("Shadow   :", shadow.shape)
print("Material :", material_cls.shape)


thresholds = {
    "SAM": {
        "shadow": compute_threshold(
            SAM_vec(shadow, shadow),
            SAM_vec(shadow, material_cls),
            higher_is_better=False
        ),
        "material": compute_threshold(
            SAM_vec(material_cls, material_cls),
            SAM_vec(material_cls, shadow),
            higher_is_better=False
        )
    },
    "SID": {
        "shadow": compute_threshold(
            SID_vec(shadow, shadow),
            SID_vec(shadow, material_cls),
            higher_is_better=False
        ),
        "material": compute_threshold(
            SID_vec(material_cls, material_cls),
            SID_vec(material_cls, shadow),
            higher_is_better=False
        )
    },
    "SCC": {
        "shadow": compute_threshold(
            SCC_vec(shadow, shadow),
            SCC_vec(shadow, material_cls),
            higher_is_better=True
        ),
        "material": compute_threshold(
            SCC_vec(material_cls, material_cls),
            SCC_vec(material_cls, shadow),
            higher_is_better=True
        )
    }
}

thr_df = pd.DataFrame(thresholds)
thr_df = thr_df.T

thr_df.to_csv(f"{material}_thresholds.csv")

print("\n✓ Thresholds saved successfully")
print(f"→ {material}_thresholds.csv")
print("\nFile structure:")
print(thr_df)