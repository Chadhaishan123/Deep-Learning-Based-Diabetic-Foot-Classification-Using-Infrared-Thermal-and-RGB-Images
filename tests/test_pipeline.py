"""Test Suite for Diabetic Foot Classification Pipeline
Verifies preprocessing, models, feature extraction, and evaluation metrics.
"""

import unittest
import numpy as np
from PIL import Image
import torch

from src.data.preprocessing import calibrate_thermal_to_temperature, pad_to_square, preprocess_image_to_tensor
from src.data.segmentation import compute_iou, compute_dice, segment_foot_mask
from src.data.feature_extraction import partition_foot_into_angiosomes, extract_120_thermal_features
from src.models.resnet import get_resnet_model
from src.models.lsnet import get_lsnet_model
from src.models.fusion_models import LF_ResNet, DE_ResNet
from src.models.tabular_models import get_tabular_model
from src.models.hybrid_ensemble import HybridLateFusionEnsemble
from src.evaluation.metrics import compute_classification_metrics


class TestDiabeticFootPipeline(unittest.TestCase):

    def test_temperature_calibration(self):
        """Verify Eq. 1: T(x, y) = T_min + (I - I_min) / (I_max - I_min) * (T_max - T_min)"""
        intensity = np.array([0, 127.5, 255], dtype=np.float32)
        temp = calibrate_thermal_to_temperature(intensity, t_min=20.0, t_max=40.0, i_min=0.0, i_max=255.0)
        self.assertAlmostEqual(temp[0], 20.0, places=2)
        self.assertAlmostEqual(temp[1], 30.0, places=2)
        self.assertAlmostEqual(temp[2], 40.0, places=2)

    def test_aspect_ratio_padding(self):
        """Verify square aspect-ratio padding to 640x640."""
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        padded = pad_to_square(img, target_size=640)
        self.assertEqual(padded.size, (640, 640))

    def test_segmentation_metrics(self):
        """Verify IoU (Eq. 2) and DICE (Eq. 3)."""
        m1 = np.ones((50, 50), dtype=np.uint8)
        m2 = np.ones((50, 50), dtype=np.uint8)
        self.assertAlmostEqual(compute_iou(m1, m2), 1.0)
        self.assertAlmostEqual(compute_dice(m1, m2), 1.0)

        # 50% overlap
        m2[:25, :] = 0
        # intersection = 25*50 = 1250, union = 2500 -> iou = 0.5
        # dice = 2*1250 / (2500 + 1250) = 2500 / 3750 = 2/3
        self.assertAlmostEqual(compute_iou(m1, m2), 0.5)
        self.assertAlmostEqual(compute_dice(m1, m2), 2.0 / 3.0)

    def test_angiosome_feature_extraction(self):
        """Verify 120 statistical feature extraction across 4 angiosomes and 2 time points."""
        temp_t0 = np.full((100, 100), 30.0, dtype=np.float32)
        mask_t0 = np.full((100, 100), 255, dtype=np.uint8)
        temp_t10 = np.full((100, 100), 28.0, dtype=np.float32)
        mask_t10 = np.full((100, 100), 255, dtype=np.uint8)

        features = extract_120_thermal_features(temp_t0, mask_t0, temp_t10, mask_t10)
        self.assertEqual(len(features), 120)

    def test_resnet_forward(self):
        """Verify ResNet-18 and ResNet-50 forward pass."""
        x = torch.randn(2, 3, 224, 224)
        m18 = get_resnet_model("resnet18", num_classes=2, pretrained=False)
        out18 = m18(x)
        self.assertEqual(out18.shape, (2, 2))

        m50 = get_resnet_model("resnet50", num_classes=4, pretrained=False)
        out50 = m50(x)
        self.assertEqual(out50.shape, (2, 4))

    def test_lsnet_forward(self):
        """Verify LSNet forward pass."""
        x = torch.randn(2, 3, 224, 224)
        lsnet_t = get_lsnet_model("lsnet-t", num_classes=2)
        out_t = lsnet_t(x)
        self.assertEqual(out_t.shape, (2, 2))

    def test_fusion_models_forward(self):
        """Verify LF-ResNet and DE-ResNet forward pass."""
        rgb = torch.randn(2, 3, 224, 224)
        ir = torch.randn(2, 3, 224, 224)

        lf = LF_ResNet(backbone="resnet18", num_classes=2, pretrained=False)
        out_lf = lf(rgb, ir)
        self.assertEqual(out_lf.shape, (2, 2))

        de = DE_ResNet(backbone="resnet18", num_classes=4, pretrained=False)
        out_de = de(rgb, ir)
        self.assertEqual(out_de.shape, (2, 4))

    def test_tabular_and_hybrid(self):
        """Verify Tabular model and Hybrid late-fusion ensemble."""
        X = np.random.randn(20, 10)
        y = np.array([0, 1] * 10)

        rf = get_tabular_model("rf", num_classes=2)
        rf.fit(X, y)
        preds = rf.predict(X)
        self.assertEqual(len(preds), 20)

        probs1 = np.full((20, 2), 0.5)
        probs2 = rf.predict_proba(X)
        ensemble = HybridLateFusionEnsemble(strategy="soft_voting", weight_cnn=0.5)
        fused = ensemble.predict(probs1, probs2)
        self.assertEqual(len(fused), 20)

    def test_evaluation_metrics(self):
        """Verify Balanced Accuracy, Precision, Recall, F1 calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_classification_metrics(y_true, y_pred, num_classes=2)
        self.assertAlmostEqual(m["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
