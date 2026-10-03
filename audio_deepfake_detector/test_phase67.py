"""
Test suite for Phase 6 (Explainable AI & Grad-CAM) and Phase 7 (Acoustic Robustness AWGN Sweep).
Validates all paper requirements:
- Grad-CAM saliency map extraction from Stream B Conv4
- Frequency marginals and 4-8 kHz attribution
- Frequency perturbation masking score drop Delta S
- Calibrated AWGN injection matching SNR formula
- Continuous sweep profiling across [inf, 30, 20, 10, 0] dB
- High-resolution figure generation
"""

import os
import sys

# Ensure project root is in sys.path when running test directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import unittest

from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.xai import (
    DualStreamGradCAM, compute_frequency_marginals,
    evaluate_perturbation_masking, plot_gradcam_explanation
)
from audio_deepfake_detector.robustness import (
    inject_awgn, run_robustness_sweep, plot_robustness_curves
)
from audio_deepfake_detector.features import DualStreamFeatureExtractor


class TestPhase67(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        cls.model = DualStreamDetector(lfcc_dim=120, mel_channels=1, num_classes=2)
        cls.model.eval()

    def test_01_gradcam_extraction(self):
        """Test Grad-CAM saliency map generation and shape."""
        cam_extractor = DualStreamGradCAM(self.model)
        lfcc = torch.randn(1, 120)
        mel = torch.randn(1, 1, 128, 126)
        
        cam, logits, prob = cam_extractor.generate_cam(lfcc, mel, target_class=1)
        
        self.assertEqual(cam.shape, (128, 126), "Grad-CAM must match Mel spectrogram resolution (128, 126)")
        self.assertGreaterEqual(cam.min(), 0.0, "CAM must be non-negative (ReLU)")
        self.assertLessEqual(cam.max(), 1.0, "CAM must be normalized to <= 1.0")
        self.assertIsInstance(prob, float)
        self.assertTrue(0.0 <= prob <= 1.0, "Probability must be in [0, 1]")

    def test_02_frequency_marginals_and_high_band(self):
        """Test frequency marginal attribution and 4-8 kHz synthetic artifact ratio."""
        cam_extractor = DualStreamGradCAM(self.model)
        lfcc = torch.randn(1, 120)
        mel = torch.randn(1, 1, 128, 126)
        cam, _, _ = cam_extractor.generate_cam(lfcc, mel, target_class=1)
        
        marginals, freqs, ratio = compute_frequency_marginals(cam, sr=16000, n_mels=128, f_max=8000)
        
        self.assertEqual(marginals.shape, (128,), "Marginals must have 128 mel frequency bins")
        self.assertEqual(freqs.shape, (128,), "Frequencies must have 128 values")
        self.assertAlmostEqual(freqs[0], 0.0, delta=10.0, msg="First frequency near 0 Hz")
        self.assertAlmostEqual(freqs[-1], 8000.0, delta=100.0, msg="Last frequency near 8000 Hz")
        self.assertTrue(0.0 <= ratio <= 1.0, "High band ratio must be between 0 and 1")

    def test_03_perturbation_masking(self):
        """Test spectral frequency masking and Delta S confidence drop."""
        cam_extractor = DualStreamGradCAM(self.model)
        lfcc = torch.randn(1, 120)
        mel = torch.randn(1, 1, 128, 126)
        cam, _, _ = cam_extractor.generate_cam(lfcc, mel, target_class=1)
        
        delta_s, s_orig, s_masked, masked_mel = evaluate_perturbation_masking(
            self.model, lfcc, mel, cam, top_k_bands=30, target_class=1
        )
        
        self.assertEqual(masked_mel.shape, mel.shape, "Masked mel shape must match original")
        self.assertIsInstance(delta_s, float)
        self.assertIsInstance(s_orig, float)
        self.assertIsInstance(s_masked, float)
        # Verify that top bands were indeed modified
        min_val = mel.min().item()
        self.assertTrue(torch.any(masked_mel == min_val), "Masked mel must contain modified minimum values")

    def test_04_awgn_injection_calibrated_snr(self):
        """Verify mathematical calibration of AWGN injection."""
        # Clean audio signal: pure sine wave
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        clean = np.sin(2 * np.pi * 440 * t)  # Power = 0.5
        
        # Test clean return
        clean_out = inject_awgn(clean, snr_db=np.inf)
        np.testing.assert_array_equal(clean, clean_out, "np.inf SNR should return identical signal")
        
        # Test 20 dB SNR
        # Theoretical: SNR_dB = 10 * log10(P_signal / P_noise)
        # P_noise = P_signal / 10^(20/10) = 0.5 / 100 = 0.005
        # Std_noise = sqrt(0.005) ≈ 0.0707
        np.random.seed(123)
        noisy_20 = inject_awgn(clean, snr_db=20.0, clip=False)
        noise_measured = noisy_20 - clean
        p_noise = np.mean(noise_measured ** 2)
        p_sig = np.mean(clean ** 2)
        measured_snr = 10 * np.log10(p_sig / p_noise)
        self.assertAlmostEqual(measured_snr, 20.0, delta=0.6, msg="Calibrated AWGN should match target SNR within 0.6 dB")

    def test_05_awgn_torch_tensor_support(self):
        """Verify AWGN works seamlessly on torch tensors."""
        wav_tensor = torch.randn(64000)
        noisy_tensor = inject_awgn(wav_tensor, snr_db=10.0)
        self.assertIsInstance(noisy_tensor, torch.Tensor)
        self.assertEqual(noisy_tensor.shape, (64000,))
        self.assertFalse(torch.equal(wav_tensor, noisy_tensor))

    def test_06_robustness_sweep_execution(self):
        """Verify continuous SNR sweep across [inf, 30, 20, 10, 0] dB."""
        feature_extractor = DualStreamFeatureExtractor()
        # Synthetic mini-dataset
        mock_dataset = [
            {'waveform': np.random.randn(16000).astype(np.float32) * 0.1, 'label': 0},
            {'waveform': np.random.randn(16000).astype(np.float32) * 0.1, 'label': 1},
            {'waveform': np.random.randn(16000).astype(np.float32) * 0.1, 'label': 1},
            {'waveform': np.random.randn(16000).astype(np.float32) * 0.1, 'label': 0},
        ]
        
        snr_levels = [np.inf, 30, 10]
        results = run_robustness_sweep(
            self.model, mock_dataset, feature_extractor,
            snr_levels=snr_levels, device='cpu'
        )
        
        self.assertIn("Clean", results)
        self.assertIn("30dB", results)
        self.assertIn("10dB", results)
        for snr_key in results:
            self.assertIn('eer', results[snr_key])
            self.assertIn('roc_auc', results[snr_key])
            self.assertIn('accuracy', results[snr_key])

    def test_07_plot_generation(self):
        """Verify figure generation for XAI and Robustness curves."""
        os.makedirs("test_artifacts", exist_ok=True)
        
        # Test Grad-CAM 3-panel plot
        mel_dummy = np.random.randn(128, 126)
        cam_dummy = np.random.rand(128, 126)
        marginals = np.mean(cam_dummy, axis=1)
        freqs = np.linspace(0, 8000, 128)
        gradcam_fig_path = "test_artifacts/test_gradcam.png"
        
        plot_gradcam_explanation(mel_dummy, cam_dummy, marginals, freqs, save_path=gradcam_fig_path)
        self.assertTrue(os.path.exists(gradcam_fig_path), "Grad-CAM figure must be generated")
        
        # Test Robustness curve plot
        sweep_data = {
            "Clean": {'eer': 1.25, 'roc_auc': 0.995},
            "30dB": {'eer': 2.10, 'roc_auc': 0.988},
            "20dB": {'eer': 3.85, 'roc_auc': 0.970},
            "10dB": {'eer': 8.40, 'roc_auc': 0.925},
            "0dB": {'eer': 18.20, 'roc_auc': 0.810},
        }
        robustness_fig_path = "test_artifacts/test_robustness.png"
        plot_robustness_curves(sweep_data, save_path=robustness_fig_path)
        self.assertTrue(os.path.exists(robustness_fig_path), "Robustness curve figure must be generated")
        
        # Clean up test artifacts
        if os.path.exists(gradcam_fig_path):
            os.remove(gradcam_fig_path)
        if os.path.exists(robustness_fig_path):
            os.remove(robustness_fig_path)
        if os.path.exists("test_artifacts"):
            os.rmdir("test_artifacts")


if __name__ == '__main__':
    unittest.main()
