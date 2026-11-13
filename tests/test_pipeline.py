"""Comprehensive Unit Test Suite for Diabetic Plantar Foot Classification
Covers Base Replication (Muscillo et al., 2026) and Novelty Extensions:
  - CMA_ResNet (Cross-Modal Attention & Gated Multimodal Fusion)
  - OrdinalFocalLoss (Clinically-Aware Ordinal Distance + Focal Loss)
  - Angiosome Thermal Asymmetry (Delta-T) & MC-Dropout Conformal Prediction
"""

import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import preprocess_image_to_tensor, calibrate_thermal_to_temperature
from src.data.segmentation import segment_foot_mask, compute_segmentation_metrics
from src.data.feature_extraction import extract_120_thermal_features
from src.models.resnet import get_resnet_model
from src.models.lsnet import get_lsnet_model
from src.models.fusion_models import LF_ResNet, DE_ResNet, CMA_ResNet
from src.models.losses import OrdinalFocalLoss
from src.models.tabular_models import get_tabular_model
from src.models.hybrid_ensemble import HybridLateFusionEnsemble
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.uncertainty import compute_angiosome_asymmetry, mc_dropout_conformal_predict


def test_aspect_ratio_padding():
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    tensor = preprocess_image_to_tensor(img)
    assert tensor.shape == (3, 224, 224)


def test_temperature_calibration():
    ir = np.array([[0, 128, 255]], dtype=np.float32)
    temp = calibrate_thermal_to_temperature(ir, t_min=24.0, t_max=36.0)
    assert np.isclose(temp[0, 0], 24.0)
    assert np.isclose(temp[0, 2], 36.0)


def test_segmentation_metrics():
    pred = np.ones((100, 100), dtype=np.uint8)
    gt = np.ones((100, 100), dtype=np.uint8)
    m = compute_segmentation_metrics(pred, gt)
    assert m["dice"] == 1.0 and m["iou"] == 1.0


def test_angiosome_feature_extraction():
    ir = np.random.uniform(25.0, 34.0, (120, 160)).astype(np.float32)
    mask = np.ones((120, 160), dtype=np.uint8)
    feats = extract_120_thermal_features(ir, ir, mask, mask)
    assert len(feats) == 120


def test_resnet_forward():
    m = get_resnet_model("resnet18", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out.shape == (2, 2)


def test_lsnet_forward():
    m = get_lsnet_model("lsnet-t", num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out.shape == (2, 4)


def test_fusion_models_forward():
    x_rgb = torch.randn(2, 3, 224, 224)
    x_ir = torch.randn(2, 3, 224, 224)
    lf = LF_ResNet(num_classes=2, pretrained=False)
    de = DE_ResNet(num_classes=2, pretrained=False)
    cma = CMA_ResNet(num_classes=4, pretrained=False)
    assert lf(x_rgb, x_ir).shape == (2, 2)
    assert de(x_rgb, x_ir).shape == (2, 2)
    assert cma(x_rgb, x_ir).shape == (2, 4)


def test_novelty_ordinal_focal_loss():
    criterion = OrdinalFocalLoss(num_classes=4, gamma=2.0, ordinal_alpha=1.5)
    logits = torch.tensor([[3.0, 0.0, -1.0, -2.0], [-2.0, -1.0, 0.0, 3.0]], requires_grad=True)
    # Near miss (true=1 vs pred=0) vs Far miss (true=3 vs pred=0)
    loss_near = criterion(logits[:1], torch.tensor([1]))
    loss_far = criterion(logits[:1], torch.tensor([3]))
    assert loss_far.item() > loss_near.item()
    loss_far.backward()
    assert logits.grad is not None


def test_novelty_uncertainty_and_angiosome_asymmetry():
    temp = np.full((100, 100), 28.0, dtype=np.float32)
    temp[:60, :50] = 31.5  # MPA hotspot (+3.5°C > 2.2°C threshold)
    mask = np.ones((100, 100), dtype=np.uint8)
    asym = compute_angiosome_asymmetry(temp, mask)
    assert asym["max_delta_t"] >= 2.2
    assert asym["thermal_asymmetry_alert"] is True
    assert asym["hottest_angiosome"] == "MPA"

    cma = CMA_ResNet(num_classes=4, pretrained=False, dropout=0.3)
    x_rgb = torch.randn(1, 3, 224, 224)
    x_ir = torch.randn(1, 3, 224, 224)
    uq = mc_dropout_conformal_predict(cma, (x_rgb, x_ir), num_samples=6, alpha=0.05)
    assert len(uq["mean_probs"]) == 4
    assert len(uq["conformal_set"]) >= 1


def test_tabular_and_hybrid():
    X = np.random.randn(20, 120).astype(np.float32)
    y = np.array([0, 1] * 10)
    rf = get_tabular_model("rf", num_classes=2)
    rf.fit(X, y)
    probs = rf.predict_proba(X[:4])
    assert probs.shape == (4, 2)
    ens = HybridLateFusionEnsemble()
    blended = ens.predict_proba(probs, probs)
    assert blended.shape == (4, 2)


def test_evaluation_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    m = compute_classification_metrics(y_true, y_pred, y_prob, num_classes=2)
    assert m["balanced_accuracy"] == 1.0

