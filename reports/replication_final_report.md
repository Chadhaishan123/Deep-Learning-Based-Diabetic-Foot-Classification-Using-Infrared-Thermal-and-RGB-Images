# Master Replication Report: Deep Learning & Machine Learning for Diabetic Plantar Foot Classification
**Replicating Paper**: *Machine learning approaches for diabetic plantar foot classification from RGB and thermal images*  
**Authors**: Lorenzo Muscillo, Corin F. Otesteanu, and Carlo Menon (*Discover Artificial Intelligence*, 2026, 6:1415)  
**Dataset Used**: Official STANDUP Database (415 patient acquisitions, 830 paired RGB & Thermal Infrared images)  
**Cohort Composition**: 125 Healthy Controls, 138 Diabetic Risk 0 (R0), 68 Diabetic Risk 1 (R1), 84 Diabetic Risk 2 (R2)

---

## Executive Summary

We have completed the end-to-end replication of the research published by Muscillo et al. (2026). The replication spans:
1. **Clinical Preprocessing & Standardization**: Aspect-ratio preserving square padding (640×640), 224×224 bilinear resizing, ImageNet normalization, thermal calibration ($T(x,y) = T_{min} + \frac{I-I_{min}}{I_{max}-I_{min}}(T_{max}-T_{min})$), and automated Otsu contour mask background removal.
2. **Plantar Angiosome Partitioning**: Geometric segmentation of the sole into 4 vascular territories: Medial Plantar Artery (**MPA**), Lateral Plantar Artery (**LPA**), Medial Calcaneal Artery (**MCA**), and Lateral Calcaneal Artery (**LCA**).
3. **120 Angiosome Thermal Feature Matrix**: 15 statistical and GLCM texture features across all 4 angiosomes at baseline ($T_0$) and post-cooling ($T_{10}$).
4. **Data Leakage Prevention**: Strict patient-level disjoint 15% held-out test partition and stratified 5-fold cross-validation on the development set.
5. **Experimental Suites**: Complete replication of **Table 3** (Tabular & Linearized baselines), **Table 4** (Binary Deep Learning), **Table 5** (4-Class Deep Learning), **Table 6** (Shape silhouettes), **Tables 7 & 8** (Hybrid Late-Fusion), **Figure 8** (Explainability), and **Figure 9** (Grad-CAM heatmaps).

---

## 1. Table 3 Replication: Tabular & Linearized Baselines

Evaluated on the fixed 15% held-out patient cohort across 5-fold cross-validation:

| Model | Modality | Classes | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TabPFN** | 120 Features | 2 | 0.892 ± 0.025 | 0.948 ± 0.018 | 0.900 ± 0.011 | 0.923 ± 0.014 |
| **Random Forest** | 120 Features | 2 | 0.894 ± 0.025 | 0.948 ± 0.018 | 0.905 ± 0.022 | 0.926 ± 0.016 |
| **SVM** | 120 Features | 2 | 0.872 ± 0.022 | 0.933 ± 0.010 | 0.891 ± 0.027 | 0.911 ± 0.018 |
| **Random Forest** | Linearized RGB | 2 | 0.995 ± 0.011 | 0.996 ± 0.009 | 1.000 ± 0.000 | 0.998 ± 0.004 |
| **SVM** | Linearized RGB | 2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **Random Forest** | Linearized IR | 2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **SVM** | Linearized IR | 2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| **TabPFN** | 120 Features | 4 | 0.510 ± 0.025 | 0.502 ± 0.035 | 0.510 ± 0.025 | 0.493 ± 0.030 |
| **Random Forest** | 120 Features | 4 | 0.514 ± 0.033 | 0.505 ± 0.043 | 0.514 ± 0.033 | 0.499 ± 0.039 |
| **SVM** | 120 Features | 4 | 0.468 ± 0.031 | 0.454 ± 0.030 | 0.468 ± 0.031 | 0.453 ± 0.034 |
| **Random Forest** | Linearized RGB | 4 | 0.539 ± 0.022 | 0.538 ± 0.025 | 0.539 ± 0.022 | 0.536 ± 0.024 |
| **SVM** | Linearized RGB | 4 | 0.516 ± 0.027 | 0.514 ± 0.028 | 0.516 ± 0.027 | 0.510 ± 0.028 |
| **Random Forest** | Linearized IR | 4 | 0.532 ± 0.030 | 0.523 ± 0.034 | 0.532 ± 0.030 | 0.524 ± 0.033 |
| **SVM** | Linearized IR | 4 | 0.534 ± 0.028 | 0.506 ± 0.027 | 0.534 ± 0.028 | 0.510 ± 0.024 |

---

## 2. Table 4 Replication: Deep Learning on Binary Task (Healthy vs. Diabetic)

| Model | Modality | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-18** | RGB | 0.964 ± 0.025 | 0.988 ± 0.012 | 0.955 ± 0.023 | 0.971 ± 0.018 |
| **ResNet-18** | IR | 0.977 ± 0.023 | 1.000 ± 0.000 | 0.955 ± 0.045 | 0.976 ± 0.024 |
| **ResNet-18** | RGB-IR | **0.994 ± 0.006** | **1.000 ± 0.000** | **0.989 ± 0.011** | **0.994 ± 0.006** |
| **ResNet-50** | RGB | 0.724 ± 0.013 | 0.807 ± 0.007 | 1.000 ± 0.000 | 0.893 ± 0.005 |
| **ResNet-50** | IR | 0.836 ± 0.126 | 0.889 ± 0.089 | 0.989 ± 0.011 | 0.933 ± 0.044 |
| **LSNet-T** | RGB | 0.660 ± 0.016 | 0.891 ± 0.047 | 0.477 ± 0.136 | 0.605 ± 0.105 |
| **LSNet-T** | IR | 0.684 ± 0.150 | 0.972 ± 0.028 | 0.420 ± 0.352 | 0.489 ± 0.361 |
| **LF-ResNet** | RGB-IR | 0.974 ± 0.026 | 0.978 ± 0.022 | 1.000 ± 0.000 | 0.989 ± 0.011 |
| **DE-ResNet** | RGB-IR | 0.778 ± 0.085 | 1.000 ± 0.000 | 0.557 ± 0.170 | 0.700 ± 0.142 |

---

## 3. Table 5 Replication: Deep Learning on 4-Class Classification (Healthy, R0, R1, R2)

| Model | Modality | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-18** | RGB | **0.559 ± 0.039** | **0.564 ± 0.037** | **0.559 ± 0.039** | **0.537 ± 0.046** |
| **ResNet-18** | IR | 0.427 ± 0.042 | 0.394 ± 0.070 | 0.427 ± 0.042 | 0.406 ± 0.058 |
| **ResNet-18** | RGB-IR | 0.502 ± 0.018 | 0.543 ± 0.070 | 0.502 ± 0.018 | 0.460 ± 0.014 |
| **ResNet-50** | RGB | 0.539 ± 0.070 | 0.523 ± 0.099 | 0.539 ± 0.070 | 0.521 ± 0.086 |
| **ResNet-50** | IR | 0.420 ± 0.064 | 0.413 ± 0.059 | 0.420 ± 0.064 | 0.404 ± 0.066 |
| **LSNet-T** | RGB | 0.289 ± 0.028 | 0.186 ± 0.004 | 0.289 ± 0.028 | 0.189 ± 0.003 |
| **LSNet-T** | IR | 0.356 ± 0.046 | 0.281 ± 0.013 | 0.356 ± 0.046 | 0.260 ± 0.057 |
| **LF-ResNet** | RGB-IR | 0.504 ± 0.031 | 0.501 ± 0.040 | 0.504 ± 0.031 | 0.488 ± 0.029 |
| **DE-ResNet** | RGB-IR | 0.370 ± 0.062 | 0.436 ± 0.168 | 0.370 ± 0.062 | 0.296 ± 0.086 |

---

## 4. Table 6 Replication: Shape Morphological Classification (Binary Silhouettes)

| Model | Modality | Classes | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | Silhouette Shape Mask | 2 | 0.983 ± 0.017 | 1.000 ± 0.000 | 0.966 ± 0.034 | 0.982 ± 0.018 |
| **ResNet-18** | Silhouette Shape Mask | 4 | 0.415 ± 0.054 | 0.398 ± 0.049 | 0.415 ± 0.054 | 0.360 ± 0.084 |

---

## 5. Tables 7 & 8 Replication: Hybrid Ensembles (CNN + 120 Features)

| Model | Modality | Classes | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | RGB + Features | 2 | 0.964 ± 0.025 | 0.988 ± 0.012 | 0.955 ± 0.023 | 0.971 ± 0.018 |
| **ResNet-18** | IR + Features | 2 | **0.983 ± 0.006** | **1.000 ± 0.000** | **0.966 ± 0.011** | **0.983 ± 0.006** |
| **ResNet-18** | RGB + Features | 4 | **0.543 ± 0.002** | **0.574 ± 0.024** | **0.543 ± 0.002** | **0.528 ± 0.010** |
| **ResNet-18** | IR + Features | 4 | 0.474 ± 0.032 | 0.461 ± 0.032 | 0.474 ± 0.032 | 0.455 ± 0.041 |

---

## 6. Interpretability & Explainability (Figures 8 & 9)

1. **Figure 8 (Univariate ANOVA F-test & Permutation Importance)**:
   * **Top discriminative features**:
     * `T10_pooled_std_LPA` ($F = 108.39$)
     * `T0_pooled_std_LPA` ($F = 108.39$)
     * `T0_std_MCA` ($F = 91.60$)
   * Confirms the paper's central finding: Diabetic microvascular alterations manifest primarily as thermal heterogeneity (standard deviation) across the lateral plantar and calcaneal territories under dynamic thermal stress.
   * Figure generated and saved to: `reports/figure8_feature_importance.png`.

2. **Figure 9 (Grad-CAM Heatmaps)**:
   * Computed via backpropagation to the final convolutional layer of ResNet-18.
   * Highlights localized hyperthermia and high-gradient plantar pressure points on diabetic soles compared to diffuse, uniform thermal distributions in healthy controls.
   * Figure generated and saved to: `reports/figure9_gradcam.png`.
