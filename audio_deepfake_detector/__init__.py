"""
Audio Deepfake Forensics Package
Implementation based on: "Towards Trustworthy Audio Deepfake Detection for Societal Good:
Spectral Evidence, Explainability, and Acoustic Robustness"
"""

__version__ = "1.0.0"

from .features import DualStreamFeatureExtractor, LFCCExtractor
from .models.dual_stream import DualStreamDetector
from .models.baselines import GMMClassifier, LogMelResNet18, LightweightRawNet2, CQCCExtractor
from .evaluate import compute_all_metrics, compute_per_attack_metrics
from .xai import DualStreamGradCAM, compute_frequency_marginals, evaluate_perturbation_masking
from .robustness import inject_awgn, run_robustness_sweep
