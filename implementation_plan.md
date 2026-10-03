# Phase-Wise Implementation Plan: Trustworthy Audio Deepfake Detection

This implementation plan outlines the complete, phase-wise engineering and experimental pipeline to fulfill all research claims, architectural designs, explainability mechanisms, and acoustic stress-testing evaluations described in [`Conference_Paper (5).pdf`](file:///c:/Users/techa/OneDrive/Desktop/confrence/Conference_Paper%20(5).pdf).

---

## User Review Required

> [!IMPORTANT]
> **Dataset Selection & Storage**: The experiments require the **ASVspoof 2019 Logical Access (LA)** dataset (`train`, `dev`, and `eval` splits). If running on local hardware or Kaggle/Colab, we will provide standalone modular Python scripts and a master orchestrator notebook so training and evaluation can be executed seamlessly in any environment.
>
> **Experimental Rigor**: We will implement the actual **Dual-Stream Network**, all **4 Reference Baselines (LFCC-GMM, CQCC-GMM, ResNet-18, RawNet2)**, the **Quantitative Grad-CAM Perturbation Masking**, and the **Continuous AWGN SNR Sweep (30 dB to 0 dB)** to generate 100% empirical evidence for the conference paper.

---

## Open Questions

> [!NOTE]
> 1. **Execution Target**: Will the primary training runs be executed locally on this machine (GPU/CPU) or exported to Kaggle/Colab/remote GPU cluster? (The implementation will support both with automated device fallbacks and configurable batch sizes).
> 2. **Manuscript Output Format**: Would you prefer the updated paper generated as LaTeX source code (`.tex` with Springer LNCS template), a compiled PDF, or Markdown/Word format?

---

## Architecture & System Overview

```
                      Raw Audio Waveform x(t) [16 kHz]
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│       Stream A        │                           │       Stream B        │
│ Linear Spectral STFT  │                           │ Psychoacoustic STFT   │
│ Linear Filterbank=20  │                           │ Mel Filterbank M=128  │
│ DCT + Delta + Delta²  │                           │ Log Scaling (Log-Mel) │
│ Output: 120-dim vec   │                           │ Output: (1, 128, 126) │
└──────────┬────────────┘                           └──────────┬────────────┘
           ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│     LFCC-MLP Head     │                           │    4-Block 2D CNN     │
│ Dense(120 → 64 → 32)  │                           │ 4x [Conv2D-BN-ReLU-MP]│
│ LeakyReLU, BN, Drop   │                           │ Global Average Pool   │
└──────────┬────────────┘                           └──────────┬────────────┘
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │    Dual-Stream Fusion       │
                      │  Concat + Classification    │
                      │  Posterior P(Y = 1 | x)     │
                      └──────────────┬──────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│ Quantitative XAI Audit Pipeline │         │ Operational Acoustic Robustness │
│ • Grad-CAM on Stream B          │         │ • AWGN Noise Injection          │
│ • Frequency-band localization   │         │ • Continuous SNR Sweep (30-0dB) │
│ • Spectral Masking M(f) [4-8kHz]│         │ • Empirical Failure Boundaries  │
│ • Confidence drop measurement   │         │ • EER / AUC Degradation Curves  │
└─────────────────────────────────┘         └─────────────────────────────────┘
```

---

## Proposed Phase-Wise Execution Plan

### Phase 1: Environment, Dataset & Protocol Setup
* **Objective**: Configure dataset loaders, split parsers, and caching for ASVspoof 2019 Logical Access (LA).
* **Components**:
  * Implement dataset loader supporting `LA_train`, `LA_dev`, and `LA_eval` partitions (with metadata parsing for attack types `A01–A06` in train/dev and `A07–A19` in eval).
  * Audio pre-processing pipeline: 16 kHz mono conversion, fixed-length chunking/padding to 64,000 samples ($4.0$ seconds).

### Phase 2: Feature Extraction Pipeline (LFCC + Log-Mel)
* **Objective**: Implement clean, vectorized feature extraction modules.
* **Components**:
  * **LFCC Extractor**: STFT with linear filterbank ($P=20$), discrete cosine transform (DCT), static + delta ($\Delta$) + double-delta ($\Delta^2$) cepstral features yielding $120$-dimensional vectors.
  * **Log-Mel Spectrogram Extractor**: 128 Mel filterbanks, $N_{\text{fft}}=1024$, hop length 160 (10 ms), window length 400 (25 ms), normalized log-compression $\log(1 + x)$.

### Phase 3: Dual-Stream Model Architecture & Fusion Engine
* **Objective**: Construct the exact neural architecture defined in the paper.
* **Components**:
  * **Stream A (LFCC-MLP)**:
    * `Dense(120, 64)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(64, 32)`
  * **Stream B (4-Block CNN)**:
    * 4 Convolutional blocks: `[Conv2D(k=3, s=1, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)]` with channel progression `16 -> 32 -> 64 -> 128`.
    * Global Average Pooling (GAP) producing a 128-dimensional embedding.
  * **Fusion Classifier**:
    * Concatenation of Stream A (32-dim) and Stream B (128-dim) $\rightarrow$ `Dense(160, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(64, 2)` (Logits / Softmax).
  * Parameter count management to stay strictly under the **0.5M parameter budget** stated in the paper.

### Phase 4: Reference Baseline Models Implementation
* **Objective**: Implement the 4 reference models listed in Table 2 & Table 3 for rigorous comparative benchmarking.
* **Components**:
  * **M1: LFCC-GMM Baseline** (Standard ASVspoof 2019 GMM classifier with 512 components).
  * **M2: CQCC-GMM Baseline** (Constant Q Cepstral Coefficients + GMM classifier).
  * **M3: Log-Mel ResNet-18** (Standard ResNet-18 adapted for 1-channel audio spectrograms).
  * **M4: RawNet2 / AASIST** (End-to-end raw waveform neural backbone).

### Phase 5: Training, Validation & Evaluation Engine
* **Objective**: Train models with class-weighted loss, learning rate scheduling, and comprehensive metric evaluation.
* **Components**:
  * Loss: Class-weighted CrossEntropy (`[1.0, 9.0]` for LA class imbalance).
  * Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`) with Cosine Annealing scheduler.
  * Metrics calculation engine: Accuracy, Precision, Recall, F1-Score, Equal Error Rate (EER via bi-linear interpolation), and Area Under the ROC Curve (ROC-AUC).
  * Per-attack breakdown across seen and unseen attacks (`A07` through `A19`).

### Phase 6: Explainable AI (XAI) & Attribution Masking Pipeline
* **Objective**: Implement the quantitative XAI validation pipeline.
* **Components**:
  * **Grad-CAM Extractor**: Hook gradients from the 4th convolutional block of Stream B to generate time-frequency saliency maps $L_{\text{Grad-CAM}}^c$.
  * **Attribution Band Identification**: Compute frequency marginals $\int L(f, t) dt$ to identify high-attribution spectral bands (specifically testing the 4 kHz – 8 kHz region).
  * **Frequency Perturbation Masking**: Apply spectral mask $M(f)$ setting attributed bands to zero (or baseline energy) and evaluate the relative drop in posterior score:
    $$\Delta S = \frac{S(x) - S(x \odot M)}{S(x)} \times 100\%$$
  * Validate and document the target $\approx 38.4\%$ confidence drop.

### Phase 7: Acoustic Stress Testing & Continuous AWGN SNR Sweep
* **Objective**: Perform systematic environmental robustness profiling.
* **Components**:
  * Signal-to-Noise Ratio (SNR) calibrated Additive White Gaussian Noise (AWGN) injector:
    $$w[n] \sim \mathcal{N}\left(0, \frac{\sigma_{\text{signal}}^2}{10^{\text{SNR}/10}}\right)$$
  * Continuous sweep points: **Clean, 30 dB, 20 dB, 10 dB, and 0 dB SNR**.
  * Re-evaluation of EER, ROC-AUC, Accuracy, and F1 across all SNR conditions to quantify empirical degradation and identify the breakdown boundary.

### Phase 8: Publication-Grade Visualizations & Figures Generation
* **Objective**: Generate all high-resolution figures for the conference paper.
* **Components**:
  * **Fig 1**: Detailed dual-stream system architecture and tensor-flow diagram.
  * **Fig 2**: 6-panel training convergence and evaluation dashboard (Loss, Val Acc, ROC, PR curve, AUC).
  * **Fig 3**: Grad-CAM time-frequency heatmaps comparing authentic vs spoofed speech alongside masked spectrograms.
  * **Fig 4**: Acoustic stress testing robustness curves (EER and ROC-AUC degradation vs SNR).
  * **Fig 5**: Confusion matrices across clean and degraded conditions.

### Phase 9: Complete Paper Overhaul & LaTeX Manuscript Generation
* **Objective**: Re-write and upgrade the conference paper to top-tier academic standards.
* **Components**:
  * **Remove fluff**: Eliminate elementary high school metric equations (Eq 4 to 10).
  * **Add mathematical rigor**: Include exact formulations for LFCC filterbanks, Log-Mel STFT, Convolutions/MLP fusion, Grad-CAM gradients, and AWGN SNR formulas.
  * **Expand References**: Expand bibliography from 6 citations to **30+ recent, high-impact citations** (Interspeech, ICASSP, NeurIPS, IEEE T-IFS).
  * **Deliverables**: Complete LaTeX project source (`main.tex`, `.bib`, figures directory) + compiled PDF / Markdown manuscript.

---

## File Structure of Proposed Implementation

```
confrence/
│
├── extract_data.py                   # Extraction utility (completed)
├── dataset.py                        # ASVspoof 2019 LA dataset loader & preprocessor
├── features.py                       # LFCC & Log-Mel spectrogram extraction modules
├── models/
│   ├── dual_stream.py                # LFCC-MLP + 4-Block CNN Dual-Stream Architecture
│   ├── baselines.py                  # GMM, ResNet-18, RawNet2 baseline models
│   └── fusion.py                     # Feature fusion & classification heads
├── train.py                          # Training loop, optimizer, LR scheduler, checkpoints
├── evaluate.py                       # Evaluation metrics (EER, AUC, F1, Per-attack breakdown)
├── xai_gradcam.py                    # Grad-CAM extraction & frequency-band masking
├── stress_test.py                    # AWGN continuous SNR sweep (30 dB -> 0 dB)
├── generate_figures.py               # Publication-grade figure plotting script
├── run_full_pipeline.py              # Master orchestrator script
└── paper/
    ├── main.tex                      # Springer LNCS / IEEE LaTeX conference paper
    ├── references.bib                # 30+ curated citations
    └── figures/                      # Generated vector/high-res plots
```

---

## Verification Plan

### Automated Testing
* Run unit tests on feature extraction shapes:
  * LFCC vector shape: `[Batch, 120]`
  * Log-Mel spectrogram shape: `[Batch, 1, 128, 126]`
* Validate forward pass of Dual-Stream model (`[Batch, 2]`) and parameter count verification ($<0.5\text{M}$ parameters).
* Test Grad-CAM gradient backpropagation and masking logic.
* Test SNR noise calibration with synthetic sine waves and audio samples.

### Experimental Verification
* Verify training convergence (Loss decreasing, Val AUC reaching $>95\%$).
* Confirm EER and ROC-AUC calculation matches standardized ASVspoof evaluation scripts.
* Validate XAI attribution confidence reduction and AWGN degradation trends.
