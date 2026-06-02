# Shadow-Invariant Material Detection Using Hyperspectral Imaging

## Overview

This project presents a shadow-invariant material detection framework using hyperspectral imagery. Traditional material classification methods often struggle in the presence of shadows because illumination variations significantly alter pixel intensities. The proposed approach leverages spectral similarity analysis to identify materials based on their spectral characteristics rather than brightness information.

By combining multiple spectral similarity metrics and a fusion-based decision strategy, the system improves material detection performance under varying lighting conditions and shadowed environments.

---

## Problem Statement

Hyperspectral images provide rich spectral information that enables accurate material identification. However, shadows introduce illumination variations that can negatively affect classification accuracy.

The goal of this project is to develop a shadow-invariant material detection framework capable of accurately separating material and shadow regions while maintaining reliable material recognition across different lighting conditions.

---

## Objectives

* Detect materials accurately under shadowed and non-shadowed conditions.
* Reduce the impact of illumination variations on material classification.
* Compare multiple spectral similarity metrics.
* Improve detection reliability through metric fusion.

---

## Dataset

The project utilizes hyperspectral imagery for both training and testing.

### Training Data

* Collection of shadow and material spectral samples.
* Manual pixel selection using pseudo-RGB visualization.
* Ground-truth generation for model development.

### Testing Data

* Independent hyperspectral scene.
* Used to evaluate model performance under unseen lighting and shadow conditions.

Dataset Source:

https://033labcodes.github.io/awesome-hyperspectral-datasets/

---

## Methodology

### Project Pipeline

#### Training Phase

1. Load hyperspectral image.
2. Select material and shadow samples.
3. Extract spectral signatures.
4. Compute reference spectral profiles.

#### Testing Phase

1. Load testing hyperspectral image.
2. Compare pixel spectra against reference signatures.
3. Apply spectral similarity metrics.
4. Generate classification masks.
5. Fuse metric outputs.
6. Produce final shadow-invariant material detection mask.

---

## Spectral Similarity Metrics

### Spectral Angle Mapper (SAM)

Measures the angular difference between spectral vectors.

Advantages:

* Robust to illumination magnitude changes.
* Widely used in hyperspectral classification.

### Spectral Information Divergence (SID)

Measures probabilistic divergence between spectral signatures.

Advantages:

* Captures spectral distribution differences.
* Effective for material discrimination.

### Spectral Correlation Coefficient (SCC)

Measures correlation between spectral shapes.

Advantages:

* Identifies spectral pattern similarity.
* Less sensitive to overall intensity scaling.

---

## Fusion-Based Classification

Individual similarity metrics may produce misclassifications in complex shadow regions.

To improve robustness, outputs from SAM, SID, and SCC are combined using a fusion strategy. The fusion process reduces false detections and improves alignment with ground-truth masks.

---

## Results

### Spectral Signature Analysis

* Material spectra exhibit relatively stable reflectance behavior.
* Shadow spectra show reduced spectral magnitude due to illumination effects.

### Classification Results

The project generates:

* SAM classification masks
* SID classification masks
* SCC classification masks
* Fusion-based classification masks
* Ground-truth comparisons
* Overlay visualizations on pseudo-RGB images

### Performance Comparison

| Method | Accuracy | Precision | Recall | F1 Score |
| ------ | -------- | --------- | ------ | -------- |
| SAM    | 0.744    | 0.001     | 0.680  | 0.003    |
| SID    | 0.740    | 0.001     | 0.680  | 0.003    |
| SCC    | 0.310    | 0.0004    | 0.902  | 0.0009   |
| Fusion | 0.784    | 0.001     | 0.741  | 0.003    |

The fusion approach achieved the highest overall accuracy among the evaluated methods.

---

## Challenges

* Spectral similarity between material and shadow regions.
* Manual sample selection process.
* Adaptive threshold tuning.
* Class imbalance between shadow and material pixels.
* Scene-dependent spectral variability.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* OpenCV
* Hyperspectral Imaging
* Spectral Analysis
* Computer Vision
* Machine Learning

---

## Future Work

* Automated sample selection.
* Deep learning-based hyperspectral classification.
* Real-time material detection systems.
* Advanced fusion and ensemble techniques.
* Improved shadow modeling and compensation.

---

## Authors

Abishai Kankipati

Shaik Sajid

IIIT Raichur

Research Supervisor:
Dr. Dubacharla Gyaneshwar
