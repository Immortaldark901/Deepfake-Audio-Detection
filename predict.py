"""
Inference & Prediction Script for Dual-Stream Audio Deepfake Detector.

Usage:
  python predict.py --audio /path/to/audio.wav
  python predict.py --demo                      # Runs on synthetic sample audio
  python predict.py --checkpoint checkpoints/dual_stream_best.pth --audio sample.wav
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.features import DualStreamFeatureExtractor
from audio_deepfake_detector.dataset import load_audio_file
from audio_deepfake_detector.xai import DualStreamGradCAM, compute_frequency_marginals, plot_gradcam_explanation


def run_prediction(model, feat_extractor, waveform, device, output_plot=None):
    """
    Runs model inference and Grad-CAM explanation on a waveform tensor.
    
    Args:
        model: DualStreamDetector
        feat_extractor: DualStreamFeatureExtractor
        waveform: 1D torch.Tensor (e.g. 64000 samples @ 16kHz)
        device: torch.device
        output_plot: optional path to save Grad-CAM explanation image
    """
    model.eval()
    feat_extractor.eval()

    wav = waveform.unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        lfcc, mel = feat_extractor(wav)

    # Prediction
    logits = model(lfcc, mel)
    probs = F.softmax(logits, dim=-1)[0]
    bonafide_prob = probs[0].item()
    spoof_prob = probs[1].item()

    is_spoof = spoof_prob >= 0.5
    label_str = "SPOOF (Deepfake)" if is_spoof else "BONA FIDE (Genuine)"
    confidence = spoof_prob if is_spoof else bonafide_prob

    print("\n" + "=" * 60)
    print("  AUDIO DEEPFAKE DETECTION RESULT")
    print("=" * 60)
    print(f"  Prediction       : {label_str}")
    print(f"  Confidence       : {confidence * 100:.2f}%")
    print(f"  Spoof Probability: {spoof_prob * 100:.2f}%")
    print(f"  Bona Fide Prob   : {bonafide_prob * 100:.2f}%")
    print("-" * 60)

    # Grad-CAM XAI explanation
    cam_extractor = DualStreamGradCAM(model)
    cam, _, _ = cam_extractor.generate_cam(lfcc, mel, target_class=1 if is_spoof else 0)
    marginals, freqs, high_ratio = compute_frequency_marginals(cam)

    print(f"  High-Freq Attribution (>4kHz): {high_ratio * 100:.2f}%")
    print("=" * 60)

    if output_plot:
        os.makedirs(os.path.dirname(output_plot) or '.', exist_ok=True)
        plot_gradcam_explanation(
            mel.squeeze().detach().cpu().numpy(),
            cam,
            marginals,
            freqs,
            title=f"Prediction: {label_str} ({confidence*100:.1f}%)",
            save_path=output_plot
        )
        print(f"  [+] Grad-CAM explanation saved to: {output_plot}\n")

    return {
        "prediction": "spoof" if is_spoof else "bonafide",
        "spoof_prob": float(spoof_prob),
        "bonafide_prob": float(bonafide_prob),
        "confidence": float(confidence),
        "high_freq_ratio": float(high_ratio)
    }


def main():
    parser = argparse.ArgumentParser(description="Run Dual-Stream Deepfake Detector Inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (.wav, .flac)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dual_stream_best.pth", help="Model checkpoint path")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic sample audio")
    parser.add_argument("--save_plot", type=str, default="prediction_gradcam.png", help="Path to save Grad-CAM plot")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cpu" if args.device == "auto" else args.device))
    print(f"Using device: {device}")

    # Initialize model and feature extractor
    model = DualStreamDetector().to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    feat_extractor = DualStreamFeatureExtractor().to(device)

    # Load audio or generate demo
    if args.audio:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        wav = load_audio_file(args.audio, target_sr=16000, max_len=64000)
        print(f"Loaded audio: {args.audio} (Length: {len(wav)} samples)")
    else:
        print("No audio specified; generating demo synthetic test sample...")
        t = torch.linspace(0, 4, 64000)
        wav = torch.sin(2 * np.pi * 440 * t) + 0.3 * torch.randn(64000)  # 440 Hz tone + noise

    run_prediction(model, feat_extractor, wav, device, output_plot=args.save_plot)


if __name__ == "__main__":
    main()
