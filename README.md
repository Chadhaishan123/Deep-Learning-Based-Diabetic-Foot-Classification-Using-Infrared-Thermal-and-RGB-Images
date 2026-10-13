# Deep Learning-Based Diabetic Foot Classification Using Infrared Thermal and RGB Images

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Discover%20AI-2026-brightgreen.svg)](https://link.springer.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Complete, rigorous replication of the research paper:
> **"Machine learning approaches for diabetic plantar foot classification from RGB and thermal images"**  
> *Lorenzo Muscillo, Corin F. Otesteanu, and Carlo Menon*  
> **Discover Artificial Intelligence** (2026) 6:1415

---

## 🌟 Overview & Scientific Contributions

Diabetic foot disease is one of the most debilitating long-term complications of diabetes mellitus, frequently culminating in neuropathic ulceration, infection, and lower-limb amputation. This repository provides an end-to-end clinical machine learning and deep learning framework to detect and classify diabetic foot conditions from paired optical (RGB) and infrared thermal (IR) plantar foot scans.

### Key Innovations Replicated:
1. **Temperature Calibration & Preprocessing**:
   $$\text{Temp}(x,y) = T_{\min} + \frac{I(x,y) - I_{\min}}{I_{\max} - I_{\min}} (T_{\max} - T_{\min})$$
   Calibrated using medical-grade infrared thermal limits ($24.0^\circ\text{C}$ to $36.0^\circ\text{C}$).
2. **Plantar Angiosome Partitioning**:
   Geometric partitioning of the sole into 4 arterial territories:
   * **MPA**: Medial Plantar Artery (inner midfoot)
   * **LPA**: Lateral Plantar Artery (outer sole)
   * **MCA**: Medial Calcaneal Artery (inner heel)
   * **LCA**: Lateral Calcaneal Artery (outer heel)
3. **120 Statistical & Texture Features**:
   Extraction of 15 metrics (Mean, Std, Median, Min, Max, Range, 25th/75th percentiles, GLCM Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) across 4 angiosomes at baseline ($T_0$) and post-cooling thermal stress ($T_{10}$).
4. **Zero Data-Leakage Patient Splits**:
   Strict patient-level disjoint 15% held-out test cohort + 5-fold stratified cross-validation on the development set.
5. **Multimodal Fusion & Explainability**:
   * **LF-ResNet**: Dual-backbone Late Fusion.
   * **DE-ResNet**: Dual-encoder ResNet with lateral residual cross-connections.
   * **Hybrid Ensemble**: Blending deep CNN representations with 120 angiosome clinical features via soft-voting.
   * **Explainability**: Univariate ANOVA F-test, Permutation Importance, and Grad-CAM class activation maps.

---

## 📊 Experimental Results (Replicated on Official STANDUP Cohort)

### Table 3: Tabular ML Baselines (120 Angiosome Features & Linearized Images)
| Model | Data / Modality | Classes | Balanced Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **TabPFN** | 120 Features | 2 | 0.892 ± 0.025 | 0.948 ± 0.018 | 0.900 ± 0.011 | 0.923 ± 0.014 |
| **Random Forest** | 120 Features | 2 | 0.894 ± 0.025 | 0.948 ± 0.018 | 0.905 ± 0.022 | 0.926 ± 0.016 |
| **SVM** | 120 Features | 2 | 0.872 ± 0.022 | 0.933 ± 0.010 | 0.891 ± 0.027 | 0.911 ± 0.018 |
| **TabPFN** | 120 Features | 4 | 0.510 ± 0.025 | 0.502 ± 0.035 | 0.510 ± 0.025 | 0.493 ± 0.030 |
| **Random Forest** | 120 Features | 4 | 0.514 ± 0.033 | 0.505 ± 0.043 | 0.514 ± 0.033 | 0.499 ± 0.039 |
| **SVM** | 120 Features | 4 | 0.468 ± 0.031 | 0.454 ± 0.030 | 0.468 ± 0.031 | 0.453 ± 0.034 |

---

### Table 4: Deep Learning Models on Binary Task (Healthy vs. Diabetic)
| Model | Modality | Balanced Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **ResNet-18** | RGB | 0.964 ± 0.025 | 0.988 ± 0.012 | 0.955 ± 0.023 | 0.971 ± 0.018 |
| **ResNet-18** | IR | 0.977 ± 0.023 | 1.000 ± 0.000 | 0.955 ± 0.045 | 0.976 ± 0.024 |
| **ResNet-18** | **RGB-IR** | **0.994 ± 0.006** | **1.000 ± 0.000** | **0.989 ± 0.011** | **0.994 ± 0.006** |
| **ResNet-50** | RGB | 0.724 ± 0.013 | 0.807 ± 0.007 | 1.000 ± 0.000 | 0.893 ± 0.005 |
| **ResNet-50** | IR | 0.836 ± 0.126 | 0.889 ± 0.089 | 0.989 ± 0.011 | 0.933 ± 0.044 |
| **LF-ResNet** | RGB-IR | 0.974 ± 0.026 | 0.978 ± 0.022 | 1.000 ± 0.000 | 0.989 ± 0.011 |
| **DE-ResNet** | RGB-IR | 0.778 ± 0.085 | 1.000 ± 0.000 | 0.557 ± 0.170 | 0.700 ± 0.142 |

---

### Table 5: Deep Learning Models on 4-Class Classification (Healthy, R0, R1, R2)
| Model | Modality | Balanced Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **ResNet-18** | **RGB** | **0.559 ± 0.039** | **0.564 ± 0.037** | **0.559 ± 0.039** | **0.537 ± 0.046** |
| **ResNet-18** | IR | 0.427 ± 0.042 | 0.394 ± 0.070 | 0.427 ± 0.042 | 0.406 ± 0.058 |
| **ResNet-18** | RGB-IR | 0.502 ± 0.018 | 0.543 ± 0.070 | 0.502 ± 0.018 | 0.460 ± 0.014 |
| **LF-ResNet** | RGB-IR | 0.504 ± 0.031 | 0.501 ± 0.040 | 0.504 ± 0.031 | 0.488 ± 0.029 |

---

### Tables 7 & 8: Hybrid Ensembles (Deep Visual Embeddings + 120 Clinical Features)
| Model | Modality | Classes | Balanced Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-18** | RGB + Features | 2 | 0.964 ± 0.025 | 0.988 ± 0.012 | 0.955 ± 0.023 | 0.971 ± 0.018 |
| **ResNet-18** | **IR + Features** | 2 | **0.983 ± 0.006** | **1.000 ± 0.000** | **0.966 ± 0.011** | **0.983 ± 0.006** |
| **ResNet-18** | **RGB + Features** | 4 | **0.543 ± 0.002** | **0.574 ± 0.024** | **0.543 ± 0.002** | **0.528 ± 0.010** |
| **ResNet-18** | IR + Features | 4 | 0.474 ± 0.032 | 0.461 ± 0.032 | 0.474 ± 0.032 | 0.455 ± 0.041 |

---

## 🔍 Explainability & Biological Findings

### Univariate ANOVA F-Scores (Figure 8)
* `T10_pooled_std_LPA` ($F = 108.39$)
* `T0_pooled_std_LPA` ($F = 108.39$)
* `T0_std_MCA` ($F = 91.60$)
* **Takeaway**: Temperature standard deviation in the **lateral plantar (LPA)** and **medial calcaneal (MCA)** angiosomes under dynamic stress is the foremost biological marker distinguishing diabetic neuropathy from healthy controls.

### Grad-CAM Activation Maps (Figure 9)
Grad-CAM heatmaps highlight high localized temperature gradient anomalies on diabetic soles, matching known plantar pressure ulceration sites.

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Chadhaishan123/Deep-Learning-Based-Diabetic-Foot-Classification-Using-Infrared-Thermal-and-RGB-Images.git
cd Deep-Learning-Based-Diabetic-Foot-Classification-Using-Infrared-Thermal-and-RGB-Images
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest Dataset & Extract Angiosome Features
```bash
python src/data/ingest_standup.py
```

### 3. Run Complete Paper Experiments
```bash
# Runs Table 3, Table 6, Tables 7 & 8, and Figure 8
python src/run_paper_experiments.py

# Runs Table 4, Table 5, and Figure 9 Grad-CAM
python src/run_deep_learning_experiments.py
```

### 4. Run Unit Test Suite
```bash
pytest tests/ -v
```

---

## 📂 Project Architecture

```
├── reports/                          # Generated markdown tables and figures
│   ├── replication_final_report.md   # Master summary report
│   ├── table3_results.md             # Baseline ML results
│   ├── table4_results.md             # Deep learning binary results
│   ├── table5_results.md             # Deep learning 4-class results
│   ├── table6_results.md             # Shape morphology results
│   ├── hybrid_results.md             # Hybrid ensemble results
│   ├── figure8_feature_importance.png# Explainability bar chart
│   └── figure9_gradcam.png           # Grad-CAM heatmap visualizations
├── src/
│   ├── config.py                     # Hyperparameters and calibration constants
│   ├── data/
│   │   ├── dataset.py                # Multi-modal PyTorch dataset loader
│   │   ├── feature_extraction.py     # 4-Angiosome 120-feature extractor
│   │   ├── preprocessing.py          # Thermal calibration and square padding
│   │   ├── segmentation.py           # Otsu background contour removal
│   │   └── split.py                  # Patient-level disjoint partitioner
│   ├── evaluation/
│   │   ├── metrics.py                # Balanced Accuracy, Precision, Recall, F1
│   │   ├── interpretability.py       # GradCAM and ANOVA F-test engine
│   │   └── reporter.py               # Markdown table builder
│   ├── models/
│   │   ├── resnet.py                 # ResNet-18 and ResNet-50 backbones
│   │   ├── lsnet.py                  # LSNet Large-Small conv architecture
│   │   ├── fusion_models.py          # LF-ResNet and DE-ResNet
│   │   ├── tabular_models.py         # TabPFN, Random Forest, and SVM
│   │   └── hybrid_ensemble.py        # Late-fusion soft voting ensemble
│   └── train.py                      # Multi-fold training engine
└── tests/
    └── test_pipeline.py              # Unit tests for preprocessing & models
```

---

## 📜 Citation
```bibtex
@article{muscillo2026machine,
  title={Machine learning approaches for diabetic plantar foot classification from RGB and thermal images},
  author={Muscillo, Lorenzo and Otesteanu, Corin F. and Menon, Carlo},
  journal={Discover Artificial Intelligence},
  volume={6},
  number={1},
  pages={1415},
  year={2026},
  publisher={Springer}
}
```
