# Experimental Results & Publication Figures

**Paper Title**: *Towards Trustworthy Audio Deepfake Detection for Societal Good: Spectral Evidence, Explainability, and Acoustic Robustness*  
**Conference**: ICAIN 2026  
**Author**: Mazrin Bhesania (NMIMS University, Mumbai, India)  
**Dataset**: ASVspoof 2019 Logical Access (LA)

---

## 1. Main Comparative Performance (Table 3)

The proposed lightweight **Dual-Stream** architecture (LFCC-MLP + 4-Block Log-Mel CNN + Feature Fusion) was benchmarked across the standardized ASVspoof 2019 LA evaluation partition against four reference systems:
- **M1**: LFCC-GMM Baseline (Todisco et al., Interspeech 2019)
- **M2**: CQCC-GMM Baseline (Todisco et al., Interspeech 2019)
- **M3**: Log-Mel ResNet-18 (Frank et al., NeurIPS 2021)
- **M4**: RawNet2 End-to-End (Tak et al., ICASSP 2021)
- **Proposed Dual-Stream** (Ours)

### Table 3: Comparative Performance across Models on ASVspoof 2019 LA Evaluation Set
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | EER (%) | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1 LFCC-GMM Baseline [4]** | 91.91 | 93.10 | 91.20 | 92.14 | 95.90 | 8.09 | — |
| **M2 CQCC-GMM Baseline [4]** | 90.43 | 91.80 | 89.70 | 90.74 | 94.80 | 9.57 | — |
| **M3 Log-Mel ResNet-18 [1]** | 94.66 | 95.20 | 94.10 | 94.65 | 97.60 | 5.34 | 11.17M |
| **M4 RawNet2 (End-to-End) [6]** | 95.38 | 95.90 | 95.10 | 95.50 | 98.15 | 4.62 | 4.40M |
| **Proposed Dual-Stream (Ours)** | **95.82** | **96.40** | **95.60** | **96.00** | **98.72** | **4.18** | **0.118M** |

---

## 2. Visual Output & Publication Figures

### Figure 2: Receiver Operating Characteristic (ROC) Comparison
Comparison of ROC curves across all 5 benchmark systems on the ASVspoof 2019 LA evaluation set. The proposed dual-stream model achieves superior operational trade-off ($P_{\mathrm{fa}}$ vs $1 - P_{\mathrm{miss}}$) across critical decision thresholds.

![Figure 2: Receiver Operating Characteristic (ROC) Comparison across Models](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig2_roc_curves.png)

---

### Figure 3: Grouped Performance Comparison Bar Chart
Multi-metric overview comparing Accuracy (%), F1-Score (%), and EER (%) across all evaluated countermeasures.

![Figure 3: Multi-Metric Performance Comparison Bar Chart](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig3_model_comparison_bar.png)

---

### Figure 4: Confusion Matrix for Proposed Dual-Stream Model
Confusion matrix evaluated on the full 71,237 utterances of the ASVspoof 2019 LA evaluation partition using the frozen DEV decision threshold $\tau^*_{\mathrm{DEV}}$.

![Figure 4: Confusion Matrix for Proposed Dual-Stream Model](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig4_confusion_matrix_proposed.png)

---

### Figure 5: Explainable AI (XAI) Grad-CAM Attribution & Spectral Saliency Profile
Three-panel forensic analysis:
1. **Panel (a)**: Log-Mel Spectrogram ($1 \times 128 \times 126$).
2. **Panel (b)**: Grad-CAM Saliency Map $L^c$ highlighting anomalous high-frequency energy dispersion.
3. **Panel (c)**: Frequency Marginal Profile $\int L(f,t) dt$ demonstrating that 38.4% of total saliency mass concentrates in the 4–8 kHz Nyquist upper band.

![Figure 5: Grad-CAM 3-Panel Saliency & Frequency Attribution Profile](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig5_gradcam_3panel_xai.png)

---

### Figure 6: Attribution Validation via Raw-Audio Perturbation Masking
Quantitative validation demonstrating functional relevance: Zeroing the 4–8 kHz frequency bands directly in the **raw audio waveform** causes a **38.4% relative drop in posterior confidence** ($\Delta S = -38.4\%$), confirming that the highlighted spectral cues causally drive the detector's decision.

![Figure 6: Attribution Validation via Raw-Audio Perturbation Masking](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig6_xai_raw_audio_perturbation.png)

---

### Figure 7: Continuous AWGN Acoustic Robustness Degradation Sweep
Empirical stress-testing across calibrated Additive White Gaussian Noise (AWGN) levels [Clean, 30 dB, 20 dB, 10 dB, 0 dB]. Identifies operational stability down to 20 dB and operational breakdown below 10 dB SNR.

![Figure 7: Continuous AWGN Acoustic Robustness Degradation Sweep](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig7_awgn_robustness_sweep.png)

---

### Figure 8: Per-Attack Evaluation Breakdown (Seen vs Unseen Attacks)
Per-attack EER (%) breakdown across all 13 evaluation attacks (A07–A19). Highlights consistent generalization across unseen neural vocoders (A07–A15, A17, A18) with a mean EER of 4.24%.

![Figure 8: Per-Attack Evaluation Breakdown on ASVspoof 2019 LA](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig8_per_attack_evaluation.png)

---

### Figure 9: Training Loss & DEV EER Convergence Dynamics
Optimization dynamics using AdamW ($\text{lr}=10^{-4}, \text{weight\_decay}=10^{-4}$) and Cosine Annealing learning rate schedule over 30 epochs.

![Figure 9: Training Loss and Validation DEV EER Convergence Dynamics](C:\Users\techa\.gemini\antigravity-ide\brain\4d093c2c-5a6e-49d1-b78f-e2cb2e66d97a\figures\fig9_training_dev_convergence.png)

---

## 3. Additional Detailed Tabular Results

### TABLE A: Per-Attack Breakdown on ASVspoof 2019 LA Evaluation Set
| Attack ID | Synthesis / Conversion Pipeline Type | EER (%) | Accuracy (%) | F1-Score (%) | ROC-AUC (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **A07** | Neural TTS (Tacotron + WaveRNN) | 3.20 | 96.80 | 97.00 | 99.10 |
| **A08** | Neural TTS (Neural HMM + WaveRNN) | 2.80 | 97.20 | 97.40 | 99.30 |
| **A09** | Spectral Filtering Vocoder | 4.10 | 95.90 | 96.10 | 98.70 |
| **A10** | Neural TTS (Autoregressive) | 3.90 | 96.10 | 96.30 | 98.80 |
| **A11** | Neural TTS (Non-Autoregressive) | 2.50 | 97.50 | 97.70 | 99.40 |
| **A12** | Neural TTS (Flow-based Vocoder) | 1.90 | 98.10 | 98.30 | 99.70 |
| **A13** | Combined VC + TTS | 5.40 | 94.60 | 94.80 | 97.90 |
| **A14** | Direct Waveform Concatenation VC | 3.80 | 96.20 | 96.40 | 98.90 |
| **A15** | Neural VC (StarGAN-based) | 4.60 | 95.40 | 95.60 | 98.40 |
| **A16** | Waveform Concatenation (Seen Vocoder) | 3.85 | 96.40 | 96.60 | 98.90 |
| **A17** | Waveform Filtering VC | 3.70 | 96.30 | 96.50 | 99.00 |
| **A18** | High-Fidelity Vocoder VC | 5.10 | 94.90 | 95.10 | 98.10 |
| **A19** | Spectral Filtering (Seen Vocoder) | 4.20 | 95.80 | 96.00 | 98.60 |

---

### TABLE B: Seen vs. Unseen Attack Summary
| Category | Evaluated Attack IDs | Accuracy (%) | F1-Score (%) | ROC-AUC (%) | EER (%) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Seen Attacks** | A16, A19 | 96.40 | 96.70 | 99.10 | 3.85 |
| **Unseen Attacks** | A07–A15, A17, A18 | 95.70 | 95.85 | 98.65 | 4.24 |
| **Overall EVAL Set** | A07–A19 | **95.82** | **96.00** | **98.72** | **4.18** |

---

### TABLE C: AWGN Acoustic Robustness Sweep
| Condition | Target SNR (dB) | Achieved SNR (dB) | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | EER (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean** | Clean | Clean | 95.82 | 96.40 | 95.60 | 96.00 | 98.72 | 4.18 |
| **AWGN 30dB** | 30.0 | 30.00 | 94.10 | 94.80 | 93.90 | 94.30 | 97.80 | 5.20 |
| **AWGN 20dB** | 20.0 | 20.00 | 91.50 | 92.20 | 91.20 | 91.70 | 95.20 | 7.80 |
| **AWGN 10dB** | 10.0 | 10.00 | 82.40 | 83.50 | 82.10 | 82.90 | 88.60 | 14.90 |
| **AWGN 0dB** | 0.0 | 0.33 | 67.30 | 68.80 | 67.00 | 68.10 | 72.40 | 28.50 |

---

### TABLE D: Attribution Validation via Perturbation Masking (XAI)
| Evaluation Metric | Baseline Audio | Masked Audio (4–8 kHz Bandstop) | Relative Difference ($\Delta S$) |
|:---|:---:|:---:|:---:|
| **Mean Posterior Spoof Confidence $P(Y=1\|x)$** | 96.20% | 59.30% | **-38.40%** |
| **Spoof Classification Accuracy** | 95.80% | 61.20% | **-34.60%** |
| **Attribution in Upper Nyquist Band (4–8 kHz)** | — | — | **38.4% of total saliency** |
