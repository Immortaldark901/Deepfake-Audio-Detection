# Towards Trustworthy Audio Deepfake Detection for Societal Good: Spectral Evidence, Explainability, and Acoustic Robustness

**Conference**: ICAIN 2026  
**Author**: Mazrin Bhesania (NMIMS University, Mumbai, India)  
**Dataset**: ASVspoof 2019 Logical Access (LA)

---

## 1. Overview & Architecture

This repository contains the complete implementation and experimental evaluation of the lightweight Dual-Stream Audio Deepfake Forensics framework.

```
                     Input Raw Audio x(t) [16 kHz, 4.0s, 64k samples]
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
             [ STREAM A ]                            [ STREAM B ]
       Linear Cepstral STFT                    Time-Frequency STFT
      P = 20 Linear Filterbank               M = 128 Mel Filterbank
      Log-Energy + DCT-II Matrix               Logarithmic Scaling
     Static + Delta + Delta-Delta                 Normalization
       Mean & Std Pooling                      1 x 128 x 126 Spectrogram
         R^120 Feature Vector                              |
                   |                                       v
                   v                              Lightweight 4-Block CNN
             LFCC-MLP                        4 x [Conv2D - BN - ReLU - MP]
        Dense (120 -> 64 -> 32)                Global Average Pooling
       LeakyReLU, BN, Dropout                  128-dim CNN Embedding
        32-dim LFCC Embedding                              |
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                            Multimodal Fusion Head
                             Concat (32 + 128 = 160)
                           Linear (160 -> 64) -> BN -> ReLU
                                Linear (64 -> 2)
                                       |
                                       v
                           Posterior Prob P(Y=1 | x)
                           (0 = Bonafide, 1 = Spoof)
```

### Parameter Budget:
- **Total Parameters**: 118,294 (< 0.5M budget)
- **Trainable Parameters**: 117,970 (~0.118M)

---

## 2. Four Pillars of the Pipeline

1. **Dual-Stream Forensic Representation**: Combines uniform linear-frequency cepstral coefficients (LFCC) preserving high-frequency vocoder cues with psychoacoustic Mel-spectrogram spatial features.
2. **Evaluation Protocol Isolation**: Optimal decision threshold $\tau^*_{\text{DEV}}$ is strictly determined on DEV and frozen for all EVAL set and robustness benchmarks.
3. **Attribution Validation via Raw-Audio Masking**: Grad-CAM saliency identifies discriminative spectral bands on Stream B; target frequency regions are filtered directly in the **RAW AUDIO WAVEFORM**, features are re-extracted, and confidence drops ($\Delta S$) are quantified.
4. **Calibrated Continuous Acoustic Stress Testing**: Injects Additive White Gaussian Noise (AWGN) across Clean, 30 dB, 20 dB, 10 dB, and 0 dB SNR, measuring actual achieved SNR and profiling error degradation.

---

## 3. Quickstart & Colab Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Dataset & Isolation
```bash
python AUDIO_DEEPFAKE_ROUTE1/verify_dataset.py
```

### Step 3: Run Full Experiment (Training, EVAL Evaluation & Baseline Benchmarks)
```bash
python AUDIO_DEEPFAKE_ROUTE1/run_experiments.py --device cuda --epochs 30
```

### Step 4: Run Explainable AI & Raw-Audio Perturbation Masking
```bash
python AUDIO_DEEPFAKE_ROUTE1/run_xai.py --checkpoint results/checkpoints/proposed_dual_stream_best.pth --num_samples 50
```

### Step 5: Run AWGN Continuous Robustness Sweep
```bash
python AUDIO_DEEPFAKE_ROUTE1/run_robustness.py --checkpoint results/checkpoints/proposed_dual_stream_best.pth
```
