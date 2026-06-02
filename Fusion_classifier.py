import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_hsi_mat(path):
    data = loadmat(path)
    cube = None
    for v in data.values():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            cube = v.astype(float)

    if cube is None:
        raise ValueError("No hyperspectral cube found")

    # Ensure (H, W, B)
    if cube.shape[0] < cube.shape[2]:
        cube = cube.transpose(1, 2, 0)

    return cube



def SAM(img, ref):
    num = np.sum(img * ref, axis=2)
    den = np.linalg.norm(img, axis=2) * np.linalg.norm(ref)
    return np.arccos(np.clip(num / (den + 1e-12), -1, 1))


def SID(img, ref):
    eps = 1e-12
    img_p = np.clip(img, eps, None)
    img_p /= img_p.sum(axis=2, keepdims=True)

    ref_p = np.clip(ref, eps, None)
    ref_p /= ref_p.sum()

    return np.sum(img_p * np.log(img_p / ref_p), axis=2) + \
           np.sum(ref_p * np.log(ref_p / img_p), axis=2)


def SCC(img, ref):
    img_c = img - img.mean(axis=2, keepdims=True)
    ref_c = ref - ref.mean()
    return np.sum(img_c * ref_c, axis=2) / (
        np.sqrt(np.sum(img_c**2, axis=2)) *
        np.sqrt(np.sum(ref_c**2)) + 1e-12
    )



material = input("Enter material name: ").strip().lower()
IMG_PATH = input("Enter test .mat image path: ").strip()
GT_PATH = input("Enter GT .mat path: ").strip()




mat_df = pd.read_csv(f"{material}_material.csv")
shd_df = pd.read_csv(f"{material}_shadow.csv")
bands = [c for c in mat_df.columns if c.startswith("band_")]

mat_ref = mat_df[bands].values.mean(axis=0)
shd_ref = shd_df[bands].values.mean(axis=0)

thr = pd.read_csv(f"{material}_thresholds.csv", index_col=0)
mat_thr = thr["material"]
shd_thr = thr["shadow"]



img = load_hsi_mat(IMG_PATH)
H, W, B = img.shape



def create_pseudo_rgb(img):
    B = img.shape[2]
    r = img[:, :, int(0.7 * B)]
    g = img[:, :, int(0.5 * B)]
    b = img[:, :, int(0.3 * B)]

    rgb = np.stack([r, g, b], axis=2)
    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-12)
    return rgb


rgb_img = create_pseudo_rgb(img)
plt.imsave(f"{material}_pseudo_rgb.png", rgb_img)



gt_data = loadmat(GT_PATH)
gt_key = [k for k in gt_data.keys() if not k.startswith("__")][0]
GT = gt_data[gt_key].astype(int)


if GT.shape != (H, W):
    if GT.T.shape == (H, W):
        print("⚠ GT transposed → fixed")
        GT = GT.T
    else:
        raise ValueError("GT size mismatch with image")

if GT.max() > 1:
    GT = (GT > 0).astype(np.uint8)

valid = np.isin(GT, [0, 1])


sam_m = SAM(img, mat_ref)
sam_s = SAM(img, shd_ref)

sid_m = SID(img, mat_ref)
sid_s = SID(img, shd_ref)

scc_m = SCC(img, mat_ref)
scc_s = SCC(img, shd_ref)


def classify(m_m, m_s, t_m, t_s, higher_better=False):
    pred = np.full((H, W), 2, dtype=np.uint8)

    if higher_better:
        pred[m_s >= t_s] = 0
        pred[m_m >= t_m] = 1
    else:
        pred[m_s <= t_s] = 0
        pred[m_m <= t_m] = 1

    return pred


pred_sam = classify(sam_m, sam_s, mat_thr["SAM"], shd_thr["SAM"])
pred_sid = classify(sid_m, sid_s, mat_thr["SID"], shd_thr["SID"])
pred_scc = classify(scc_m, scc_s, mat_thr["SCC"], shd_thr["SCC"], higher_better=True)



fusion = np.full((H, W), 2, dtype=np.uint8)

fusion[
    (sam_m <= mat_thr["SAM"]) &
    (sid_m <= mat_thr["SID"]) &
    (scc_m >= mat_thr["SCC"])
] = 1

fusion[
    (sam_s <= shd_thr["SAM"]) &
    (sid_s <= shd_thr["SID"]) &
    (scc_s >= shd_thr["SCC"])
] = 0



def eval_metrics(name, pred):
    mask = np.isin(pred, [0, 1]) & valid
    gt_eval = GT[mask]
    pr_eval = pred[mask]

    print(f"\n[{name}]")
    print("Accuracy :", accuracy_score(gt_eval, pr_eval))
    print("Precision:", precision_score(gt_eval, pr_eval, zero_division=0))
    print("Recall   :", recall_score(gt_eval, pr_eval, zero_division=0))
    print("F1 Score :", f1_score(gt_eval, pr_eval, zero_division=0))


print("\n===== PERFORMANCE METRICS =====")
eval_metrics("SAM", pred_sam)
eval_metrics("SID", pred_sid)
eval_metrics("SCC", pred_scc)
eval_metrics("FUSION", fusion)



def save_mask(name, mask):
    plt.imsave(name, mask, cmap="gray")

save_mask(f"{material}_SAM_mask.png", pred_sam)
save_mask(f"{material}_SID_mask.png", pred_sid)
save_mask(f"{material}_SCC_mask.png", pred_scc)
save_mask(f"{material}_FUSION_mask.png", fusion)


def overlay_mask(rgb, mask, alpha=0.5):
    overlay = rgb.copy()

    overlay[mask == 1] = (1 - alpha) * overlay[mask == 1] + alpha * np.array([0, 1, 0])
    overlay[mask == 0] = (1 - alpha) * overlay[mask == 0] + alpha * np.array([0, 0, 1])

    return overlay


overlay_img = overlay_mask(rgb_img, fusion)
plt.imsave(f"{material}_fusion_overlay.png", overlay_img)



def save_gt_vs_pred(name, pred):
    vis = np.zeros((H, W, 3), dtype=float)

    vis[(GT == 0) & (pred == 0)] = [0, 0, 1]   # TN
    vis[(GT == 1) & (pred == 1)] = [0, 1, 0]   # TP
    vis[(GT == 0) & (pred == 1)] = [1, 0, 0]   # FP
    vis[(GT == 1) & (pred == 0)] = [1, 1, 0]   # FN

    plt.imsave(name, vis)


save_gt_vs_pred(f"{material}_GT_vs_FUSION.png", fusion)

print("\n✓ ALL outputs saved successfully")