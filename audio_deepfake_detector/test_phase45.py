"""
Phase 4-5 Automated Test Suite.
Validates:
  - All baseline model architectures (CQCC, GMM, ResNet-18, RawNet2)
  - Evaluation metrics engine (EER, ROC-AUC, per-attack breakdown)
  - Training engine (class weights, optimizer, loss, backward pass)
  - Full demo pipeline end-to-end (synthetic data)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from audio_deepfake_detector.models.baselines import (
    GMMClassifier, LogMelResNet18, LightweightRawNet2, CQCCExtractor
)
from audio_deepfake_detector.evaluate import (
    compute_eer, compute_all_metrics, compute_per_attack_metrics,
    format_metrics_table, format_per_attack_table
)
from audio_deepfake_detector.train import get_class_weights
from audio_deepfake_detector.dataset import SyntheticDemoDataset, get_dataloader
from audio_deepfake_detector.features import DualStreamFeatureExtractor
from audio_deepfake_detector.models.dual_stream import DualStreamDetector

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(cond, msg):
    status = PASS if cond else FAIL
    print(f"  {status}  {msg}")
    return cond

results = []

print("\n" + "="*65)
print("  PHASE 4-5 TEST SUITE: Baselines, Training, Evaluation")
print("="*65)

# ─────────────────────────────────────────────────
# TEST 1: CQCC Extractor (M2 baseline front-end)
# ─────────────────────────────────────────────────
print("\n[1] CQCC Extractor (M2 CQCC-GMM baseline front-end)")
cqcc = CQCCExtractor(sample_rate=16000, n_bins=20, n_ceps=20)
dummy = torch.randn(4, 64000)
out   = cqcc(dummy)
results.append(check(out.shape == (4, 120), f"CQCC output shape=(4,120) -> {out.shape}"))
results.append(check(not torch.isnan(out).any(), "No NaN in CQCC output"))
results.append(check(not torch.isinf(out).any(), "No Inf in CQCC output"))

# ─────────────────────────────────────────────────
# TEST 2: GMM Classifier (M1/M2)
# ─────────────────────────────────────────────────
print("\n[2] GMM Classifier (M1 LFCC-GMM / M2 CQCC-GMM baselines)")
gmm = GMMClassifier(n_components=4, covariance_type='diag', max_iter=10)

np.random.seed(42)
X_bona  = np.random.randn(100, 120).astype(np.float32)
X_spoof = np.random.randn(100, 120).astype(np.float32) + 1.0
gmm.fit(X_bona, X_spoof, verbose=False)

X_test  = np.vstack([X_bona[:20], X_spoof[:20]])
scores  = gmm.score(X_test)
preds   = gmm.predict(X_test)

results.append(check(scores.shape == (40,),       f"GMM score output shape=(40,) -> {scores.shape}"))
results.append(check(((scores >= 0) & (scores <= 1)).all(), f"GMM scores in [0,1]: min={scores.min():.3f} max={scores.max():.3f}"))
results.append(check(preds.shape == (40,),        f"GMM predictions shape=(40,)"))
results.append(check(set(preds).issubset({0,1}),  f"GMM predictions are binary {{0,1}}"))

# ─────────────────────────────────────────────────
# TEST 3: Log-Mel ResNet-18 (M3)
# ─────────────────────────────────────────────────
print("\n[3] M3: Log-Mel ResNet-18")
resnet = LogMelResNet18(num_classes=2)
mel_in = torch.randn(4, 1, 128, 126)
resnet.eval()
with torch.no_grad():
    logit = resnet(mel_in)
params = resnet.count_parameters()
results.append(check(logit.shape == (4, 2), f"ResNet-18 output=(4,2) -> {logit.shape}"))
results.append(check(params > 0,             f"ResNet-18 parameter count: {params:,}"))
print(f"   ResNet-18 total params: {params:,}")

# ─────────────────────────────────────────────────
# TEST 4: LightweightRawNet2 (M4)
# ─────────────────────────────────────────────────
print("\n[4] M4: Lightweight RawNet2")
rawnet = LightweightRawNet2(num_classes=2, sinc_channels=64, gru_hidden=128)
wav_in = torch.randn(4, 64000)
rawnet.eval()
with torch.no_grad():
    logit = rawnet(wav_in)
params = rawnet.count_parameters()
results.append(check(logit.shape == (4, 2), f"RawNet2 output=(4,2) -> {logit.shape}"))
results.append(check(params > 0,             f"RawNet2 parameter count: {params:,}"))
print(f"   RawNet2 total params: {params:,}")

# ─────────────────────────────────────────────────
# TEST 5: Evaluation Metrics Engine
# ─────────────────────────────────────────────────
print("\n[5] Evaluation Metrics Engine (EER, ROC-AUC, Accuracy, F1)")

np.random.seed(0)
n = 1000
true_labels = np.array([0]*500 + [1]*500)
# Good detector: bonafide gets low scores, spoof gets high scores
good_scores = np.concatenate([
    np.random.beta(2, 8, 500),   # bonafide: low scores
    np.random.beta(8, 2, 500)    # spoof: high scores
])

eer, eer_thresh = compute_eer(true_labels, good_scores)
metrics = compute_all_metrics(true_labels, good_scores)

results.append(check(0.0 <= eer <= 0.3,         f"EER in [0,0.3] for decent detector: {eer*100:.2f}%"))
results.append(check(metrics['accuracy'] > 0.7,  f"Accuracy > 70%: {metrics['accuracy']*100:.2f}%"))
results.append(check(metrics['roc_auc'] > 0.85,  f"ROC-AUC > 85%: {metrics['roc_auc']*100:.2f}%"))
results.append(check(0.0 <= metrics['pfa'] <= 1.0,   f"P_fa in [0,1]: {metrics['pfa']*100:.2f}%"))
results.append(check(0.0 <= metrics['pmiss'] <= 1.0, f"P_miss in [0,1]: {metrics['pmiss']*100:.2f}%"))
results.append(check(metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'] == n,
                     f"TP+TN+FP+FN = {metrics['tp']+metrics['tn']+metrics['fp']+metrics['fn']} = N={n}"))
print(format_metrics_table(metrics, "Sample Metrics Check"))

# ─────────────────────────────────────────────────
# TEST 6: Per-Attack Breakdown (A07-A19)
# ─────────────────────────────────────────────────
print("\n[6] Per-Attack Breakdown (simulated A07-A19)")
np.random.seed(1)
n_per_atk  = 50
attacks    = [f"A{i:02d}" for i in range(7, 20)]
atk_ids    = ['-']*500 + [a for a in attacks for _ in range(n_per_atk)]
lbl_arr    = np.array([0]*500 + [1]*(len(attacks)*n_per_atk))
scr_arr    = np.concatenate([
    np.random.beta(2, 8, 500),
    np.random.beta(7, 3, len(attacks)*n_per_atk)
])

per_atk = compute_per_attack_metrics(lbl_arr, scr_arr, np.array(atk_ids))
results.append(check(len(per_atk) == len(attacks), f"Per-attack results for all {len(attacks)} attacks"))
results.append(check(all(0.0 <= v['eer'] <= 1.0 for v in per_atk.values()), "All per-attack EERs in [0,1]"))
print(format_per_attack_table(per_atk, "Simulated A07-A19 Per-Attack EER"))

# ─────────────────────────────────────────────────
# TEST 7: Class Weights Computation
# ─────────────────────────────────────────────────
print("\n[7] Class Weights (ASVspoof 2019 LA imbalance handling)")
# LA train: ~22800 spoof, ~2580 bonafide
labels_simulated = [0]*2580 + [1]*22800
weights = get_class_weights(labels_simulated)
results.append(check(weights[0] > weights[1], f"Bonafide weight > Spoof weight (minority upweighted): {weights.tolist()}"))
results.append(check(weights.shape == (2,),    f"Weights shape=(2,)"))

# ─────────────────────────────────────────────────
# TEST 8: End-to-End Training Step (Dual-Stream on Synthetic Data)
# ─────────────────────────────────────────────────
print("\n[8] End-to-End Training Step (Dual-Stream + Class-Weighted Loss)")
device   = torch.device('cpu')
feat_ext = DualStreamFeatureExtractor()
model    = DualStreamDetector()
model.train()

dataset = SyntheticDemoDataset(num_samples=16, max_len=64000)
loader  = get_dataloader(dataset, batch_size=8)

class_wts = torch.tensor([3.0, 1.0])  # bonafide upweighted
criterion = torch.nn.CrossEntropyLoss(weight=class_wts)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

batch = next(iter(loader))
wav   = batch['waveform']
lbls  = batch['label']

with torch.no_grad():
    lfcc, mel = feat_ext(wav)

optimizer.zero_grad()
logits = model(lfcc, mel)
loss   = criterion(logits, lbls)
loss.backward()
optimizer.step()

results.append(check(loss.item() > 0,     f"Training loss > 0: {loss.item():.4f}"))
results.append(check(logits.shape == (8, 2), f"Output logits shape=(8,2)"))
grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
results.append(check(grad_ok, "All trainable parameters received gradients"))

# ─────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
failed = total - passed

print("\n" + "="*65)
print(f"  PHASE 4-5 SUMMARY: {passed}/{total} checks passed")
print("="*65)
if failed == 0:
    print("  >>> PHASE 4-5 FULLY VERIFIED - READY FOR TRAINING <<<")
else:
    print(f"  WARNING: {failed} check(s) FAILED")
print("="*65 + "\n")
