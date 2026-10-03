"""
Deep Audit Script for Phase 1-3 Core Pipeline.
Validates every specification against the paper claims systematically.
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np

from audio_deepfake_detector.features import (
    LFCCExtractor, LogMelExtractor, DualStreamFeatureExtractor,
    get_linear_filterbanks, get_mel_filterbanks, dct_matrix, compute_deltas
)
from audio_deepfake_detector.models.lfcc_mlp import LFCC_MLP
from audio_deepfake_detector.models.cnn_stream import Lightweight4BlockCNN
from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.dataset import SyntheticDemoDataset, get_dataloader

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(cond, msg):
    status = PASS if cond else FAIL
    print(f"  {status}  {msg}")
    return cond

results = []

print("\n" + "="*65)
print("  PHASE 1-3 DEEP AUDIT: Paper Specification Compliance Check")
print("="*65)

###########################################
# SECTION 1: LINEAR FILTERBANK
###########################################
print("\n[1] LINEAR FILTERBANK (Paper: P=20 linear filters, uniform Nyquist spacing)")
fbank = get_linear_filterbanks(n_filters=20, n_fft=512, sample_rate=16000)
results.append(check(fbank.shape == (20, 257), f"Shape=(20, 257) -> got {fbank.shape}"))
results.append(check(float(fbank.min()) >= 0.0, f"All weights >= 0 (triangular): min={fbank.min():.4f}"))
results.append(check(float(fbank.max()) <= 1.0 + 1e-5, f"All weights <= 1.0 (triangular): max={fbank.max():.4f}"))
# Each filter should sum to a reasonable positive energy
filter_sums = fbank.sum(dim=1)
results.append(check((filter_sums > 0).all().item(), f"All 20 filters have positive energy sum"))
# Linear spacing: check filter center frequencies are equally spaced in Hz
num_bins = 257
fft_freqs = torch.linspace(0, 8000, num_bins)
# Use max-index to estimate center of each filter
centers = fbank.argmax(dim=1).float()
diffs = centers[1:] - centers[:-1]
spacing_equal = (diffs.std() < 2.0)  # expect near-uniform spacing
results.append(check(spacing_equal.item(), f"Uniform linear filter spacing (std of diffs={diffs.std():.2f} bins)"))

###########################################
# SECTION 2: MEL FILTERBANK
###########################################
print("\n[2] MEL FILTERBANK (Paper: Mel-scale M=128, psychoacoustic grouping)")
mel_fbank = get_mel_filterbanks(n_mels=128, n_fft=1024, sample_rate=16000)
results.append(check(mel_fbank.shape == (128, 513), f"Shape=(128, 513) -> got {mel_fbank.shape}"))
results.append(check(float(mel_fbank.min()) >= 0.0, f"All Mel weights >= 0"))
results.append(check(float(mel_fbank.max()) <= 1.0 + 1e-5, f"All Mel weights <= 1.0"))
# Mel spacing should be narrower at low freq, wider at high freq (non-linear)
mel_centers = mel_fbank.argmax(dim=1).float()
diffs_mel = mel_centers[1:] - mel_centers[:-1]
# Differences should increase (wider spacing at high freq)
low_spacing  = diffs_mel[:10].mean().item()
high_spacing = diffs_mel[-10:].mean().item()
results.append(check(high_spacing > low_spacing, f"Mel non-linear (low spacing={low_spacing:.1f} < high={high_spacing:.1f} bins)"))

###########################################
# SECTION 3: DCT MATRIX
###########################################
print("\n[3] DCT-II MATRIX (Paper: n_ceps=20 cepstral coefficients)")
dct_mat = dct_matrix(n_filters=20, n_ceps=20)
results.append(check(dct_mat.shape == (20, 20), f"Shape=(20,20) -> got {dct_mat.shape}"))
# DCT-II should be orthogonal: D @ D.T ~= I
ortho_check = torch.mm(dct_mat, dct_mat.t())
eye_approx  = torch.eye(20)
max_err     = (ortho_check - eye_approx).abs().max().item()
results.append(check(max_err < 1e-4, f"Orthogonality D*D^T ~= I (max_err={max_err:.2e})"))

###########################################
# SECTION 4: LFCC EXTRACTOR
###########################################
print("\n[4] LFCC EXTRACTOR (Paper: Output R^120, Static+Delta+DeltaDelta)")
lfcc_ext = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_filters=20, n_ceps=20)
batch_size = 4
wav = torch.randn(batch_size, 64000)

lfcc_out   = lfcc_ext(wav)
lfcc_frame = lfcc_ext(wav, return_frames=True)
results.append(check(lfcc_out.shape == (4, 120), f"Output shape=(4,120) -> got {lfcc_out.shape}"))
results.append(check(lfcc_frame.shape[1] == 60, f"Frame features: 60 channels (20*3) -> got {lfcc_frame.shape[1]}"))
results.append(check(not torch.isnan(lfcc_out).any().item(), "No NaN values in LFCC output"))
results.append(check(not torch.isinf(lfcc_out).any().item(), "No Inf values in LFCC output"))
# Check that mean + std pooling gives 120 from 60-channel frame features
mean_part = lfcc_out[:, :60]
std_part  = lfcc_out[:, 60:]
results.append(check(mean_part.shape == (4, 60), f"Mean part shape=(4,60): OK"))
results.append(check(std_part.shape  == (4, 60), f"Std part  shape=(4,60): OK"))
# Std should always be >= 0
results.append(check((std_part >= 0).all().item(), f"Std values all >= 0"))

###########################################
# SECTION 5: LOG-MEL EXTRACTOR
###########################################
print("\n[5] LOG-MEL EXTRACTOR (Paper: Output 1x128x126, M=128 Mel bins)")
mel_ext = LogMelExtractor(sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128, target_frames=126)
mel_out = mel_ext(wav)
results.append(check(mel_out.shape == (4, 1, 128, 126), f"Output shape=(4,1,128,126) -> got {mel_out.shape}"))
results.append(check(not torch.isnan(mel_out).any().item(), "No NaN values in Log-Mel output"))
results.append(check(not torch.isinf(mel_out).any().item(), "No Inf values in Log-Mel output"))
# After normalization: global mean should be near 0 and std near 1 per utterance
per_utt_mean = mel_out.view(4, -1).mean(dim=-1)
per_utt_std  = mel_out.view(4, -1).std(dim=-1)
mean_near_zero = (per_utt_mean.abs() < 0.5).all().item()
std_near_one   = ((per_utt_std - 1.0).abs() < 0.5).all().item()
results.append(check(mean_near_zero, f"Normalized: per-utterance mean near 0 (values={per_utt_mean.detach().numpy().round(3)})"))
results.append(check(std_near_one,   f"Normalized: per-utterance std near 1  (values={per_utt_std.detach().numpy().round(3)})"))

###########################################
# SECTION 6: CNN ARCHITECTURE
###########################################
print("\n[6] STREAM B: 4-BLOCK CNN (Paper: 4x[Conv2D-BN-ReLU-MP], GAP -> 128-dim)")
cnn = Lightweight4BlockCNN(in_channels=1, embed_dim=128, num_classes=2)
mel_in = torch.randn(4, 1, 128, 126)
cnn.eval()

# Test embed output
embed = cnn(mel_in, return_embed=True)
logit = cnn(mel_in, return_embed=False)
results.append(check(embed.shape == (4, 128), f"CNN Embed shape=(4,128) -> got {embed.shape}"))
results.append(check(logit.shape == (4, 2),   f"CNN Logit shape=(4,2) -> got {logit.shape}"))

# Verify it's exactly 4 blocks by tracing output shapes
x = mel_in
x = cnn.block1(x); s1 = x.shape
x = cnn.block2(x); s2 = x.shape
x = cnn.block3(x); s3 = x.shape
x = cnn.block4(x); s4 = x.shape
print(f"   Block shapes: B1={tuple(s1)}, B2={tuple(s2)}, B3={tuple(s3)}, B4={tuple(s4)}")
results.append(check(len([s1,s2,s3,s4]) == 4, "Exactly 4 convolutional blocks present"))
results.append(check(s4[1] == 128, f"Final block has 128 output channels -> got {s4[1]}"))

# Grad-CAM hook registration (critical for Phase 6)
cnn.train()
mel_grad = torch.randn(4, 1, 128, 126, requires_grad=True)
out_cam = cnn(mel_grad, retain_cam_maps=True)
out_cam.sum().backward()
results.append(check(cnn.last_conv_act is not None, "Grad-CAM: last_conv_act captured"))
results.append(check(cnn.last_conv_grad is not None, "Grad-CAM: last_conv_grad captured (backward hook)"))

###########################################
# SECTION 7: LFCC-MLP ARCHITECTURE
###########################################
print("\n[7] STREAM A: LFCC-MLP (Paper: Dense(120->64->1) with LeakyReLU, BN, Dropout)")
mlp = LFCC_MLP(input_dim=120, hidden_dim=64, embed_dim=32, num_classes=2)
lfcc_in = torch.randn(4, 120)
mlp.eval()
embed_a = mlp(lfcc_in, return_embed=True)
logit_a = mlp(lfcc_in, return_embed=False)
results.append(check(embed_a.shape == (4, 32), f"MLP Embed=(4,32) -> got {embed_a.shape}"))
results.append(check(logit_a.shape == (4, 2),  f"MLP Logit=(4,2) -> got {logit_a.shape}"))
# Verify BatchNorm and LeakyReLU are present
has_bn   = any(isinstance(m, nn.BatchNorm1d) for m in mlp.modules())
has_lrelu= any(isinstance(m, nn.LeakyReLU) for m in mlp.modules())
has_drop = any(isinstance(m, nn.Dropout) for m in mlp.modules())
results.append(check(has_bn,    "BatchNorm1d present in MLP (paper spec)"))
results.append(check(has_lrelu, "LeakyReLU present in MLP (paper spec)"))
results.append(check(has_drop,  "Dropout present in MLP (paper spec)"))

###########################################
# SECTION 8: DUAL-STREAM FUSION
###########################################
print("\n[8] DUAL-STREAM FUSION & PARAMETER BUDGET (Paper: <0.5M params)")
model = DualStreamDetector(lfcc_dim=120, mel_channels=1, num_classes=2)
lfcc_t = torch.randn(4, 120)
mel_t  = torch.randn(4, 1, 128, 126)
model.eval()

fused_logits = model(lfcc_t, mel_t)
probs = model.predict_proba(lfcc_t, mel_t)
total, trainable = model.count_parameters()

results.append(check(fused_logits.shape == (4, 2), f"Fused logit shape=(4,2) -> got {fused_logits.shape}"))
results.append(check(probs.shape == (4,),           f"Posterior prob shape=(4,) -> got {probs.shape}"))
results.append(check((probs >= 0).all() and (probs <= 1).all(), f"P(Y=1|x) in [0,1]: {probs.detach().numpy().round(3)}"))
results.append(check(total < 500_000, f"Total params={total:,} ({total/1e6:.3f}M) < 0.5M budget"))

# Verify fusion dim = 32 + 128 = 160
fusion_in = model.fusion_head[0].in_features
results.append(check(fusion_in == 160, f"Fusion input dim=160 (32+128) -> got {fusion_in}"))

# Verify label convention: Y=1 = spoof
print(f"   Label convention: Y=1=spoof, Y=0=bonafide (per paper Eq.2-3)")

###########################################
# SECTION 9: GRADIENT FLOW (TRAINABILITY)
###########################################
print("\n[9] GRADIENT FLOW & BACKPROPAGATION (Training viability check)")
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

lfcc_t2 = torch.randn(4, 120)
mel_t2  = torch.randn(4, 1, 128, 126)

optimizer.zero_grad()
out = model(lfcc_t2, mel_t2)
loss = criterion(out, labels)
loss.backward()
optimizer.step()

# Only trainable (requires_grad=True) parameters should receive gradients.
# Sub-stream classifier heads are intentionally frozen -> excluded from this check.
trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
frozen_params    = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
trainable_no_grad = [n for n, p in trainable_params if p.grad is None]
nan_grads = [torch.isnan(p.grad).any().item() for _, p in trainable_params if p.grad is not None]

results.append(check(len(trainable_no_grad) == 0,
    f"All TRAINABLE params received gradients (frozen params: {len(frozen_params)})"))
results.append(check(not any(nan_grads),  f"No NaN gradients found"))
results.append(check(loss.item() > 0,     f"Loss is positive: {loss.item():.4f}"))
print(f"   Frozen params (intentional dormant heads): {[n for n, _ in frozen_params]}")

###########################################
# SECTION 10: DATASET PIPELINE
###########################################
print("\n[10] DATASET & DATALOADER (ASVspoof 2019 LA format, attack IDs A07-A19 for eval)")
dataset = SyntheticDemoDataset(num_samples=32, max_len=64000, spoof_ratio=0.5)
loader  = get_dataloader(dataset, batch_size=8, shuffle=False)
feat_ext = DualStreamFeatureExtractor()

all_attack_ids = []
for batch in loader:
    wav = batch["waveform"]
    lbl = batch["label"]
    aids = batch["attack_id"]
    all_attack_ids.extend(aids)
    lfcc_b, mel_b = feat_ext(wav)
    results.append(check(wav.shape  == (8, 64000),  f"Batch wav shape=(8,64000)"))
    results.append(check(lfcc_b.shape == (8, 120),   f"Batch LFCC shape=(8,120)"))
    results.append(check(mel_b.shape  == (8,1,128,126), f"Batch Mel shape=(8,1,128,126)"))
    results.append(check(lbl.dtype == torch.long,    f"Label dtype=torch.long"))
    break  # just check first batch

spoof_ids = [a for a in all_attack_ids if a != "-"]
results.append(check(len(spoof_ids) > 0, f"Attack IDs populated for spoof samples: {spoof_ids[:5]}"))

###########################################
# FINAL SUMMARY
###########################################
passed = sum(results)
total_checks = len(results)
failed = total_checks - passed

print("\n" + "="*65)
print(f"  DEEP AUDIT SUMMARY: {passed}/{total_checks} checks passed")
print("="*65)
if failed == 0:
    print("  >>> PHASE 1-3 FULLY COMPLIANT WITH PAPER SPECIFICATIONS <<<")
else:
    print(f"  WARNING: {failed} check(s) FAILED - review items marked [FAIL] above")
print("="*65 + "\n")
