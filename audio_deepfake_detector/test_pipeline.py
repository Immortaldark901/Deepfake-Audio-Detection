"""
Automated Test & Validation Suite for Phase 1-3 Core Pipeline.
Tests:
1. Vectorized LFCC (120-dim) & Log-Mel (1 x 128 x 126) feature extractors.
2. Lightweight 4-Block CNN and LFCC-MLP backbones.
3. Dual-Stream Network architecture, parameter budget (<0.5M), and gradient flow.
4. Dataset loaders, batch collation, and synthetic verification pipeline.
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_deepfake_detector.features import LFCCExtractor, LogMelExtractor, DualStreamFeatureExtractor
from audio_deepfake_detector.models.lfcc_mlp import LFCC_MLP
from audio_deepfake_detector.models.cnn_stream import Lightweight4BlockCNN
from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.dataset import SyntheticDemoDataset, get_dataloader


def test_feature_extractors():
    print("\n" + "=" * 60)
    print(" [TEST 1] Feature Extraction Verification (LFCC & Log-Mel)")
    print("=" * 60)
    
    batch_size = 4
    samples = 64000  # 4 seconds @ 16kHz
    dummy_audio = torch.randn(batch_size, samples)
    
    # 1. LFCC Extractor
    lfcc_ext = LFCCExtractor(sample_rate=16000, n_filters=20, n_ceps=20)
    lfcc_out = lfcc_ext(dummy_audio)
    print(f" [+] LFCC Input shape       : {dummy_audio.shape}")
    print(f" [+] LFCC Output shape      : {lfcc_out.shape} (Expected: ({batch_size}, 120))")
    assert lfcc_out.shape == (batch_size, 120), f"Invalid LFCC shape: {lfcc_out.shape}"
    assert not torch.isnan(lfcc_out).any(), "NaN values found in LFCC output!"
    print(" [OK] LFCC Extractor PASSED.")
    
    # 2. Log-Mel Extractor
    mel_ext = LogMelExtractor(sample_rate=16000, n_mels=128, target_frames=126)
    mel_out = mel_ext(dummy_audio)
    print(f" [+] Log-Mel Output shape   : {mel_out.shape} (Expected: ({batch_size}, 1, 128, 126))")
    assert mel_out.shape == (batch_size, 1, 128, 126), f"Invalid Log-Mel shape: {mel_out.shape}"
    assert not torch.isnan(mel_out).any(), "NaN values found in Log-Mel output!"
    print(" [OK] Log-Mel Extractor PASSED.")
    
    # 3. Dual-Stream Combined Extractor
    dual_ext = DualStreamFeatureExtractor(sample_rate=16000)
    lfcc_feat, mel_feat = dual_ext(dummy_audio)
    assert lfcc_feat.shape == (batch_size, 120)
    assert mel_feat.shape == (batch_size, 1, 128, 126)
    print(" [OK] Combined Dual-Stream Feature Extractor PASSED.")


def test_model_architectures():
    print("\n" + "=" * 60)
    print(" [TEST 2] Model Architectures & Parameter Budget Verification")
    print("=" * 60)
    
    batch_size = 4
    lfcc_tensor = torch.randn(batch_size, 120)
    mel_tensor = torch.randn(batch_size, 1, 128, 126)
    
    # 1. Stream A (LFCC-MLP)
    mlp = LFCC_MLP(input_dim=120, hidden_dim=64, embed_dim=32, num_classes=2)
    mlp_logits = mlp(lfcc_tensor, return_embed=False)
    mlp_embed = mlp(lfcc_tensor, return_embed=True)
    print(f" [+] Stream A Logits shape  : {mlp_logits.shape} (Expected: ({batch_size}, 2))")
    print(f" [+] Stream A Embed shape   : {mlp_embed.shape} (Expected: ({batch_size}, 32))")
    assert mlp_logits.shape == (batch_size, 2)
    assert mlp_embed.shape == (batch_size, 32)
    print(" [OK] Stream A (LFCC-MLP) PASSED.")
    
    # 2. Stream B (4-Block CNN)
    cnn = Lightweight4BlockCNN(in_channels=1, embed_dim=128, num_classes=2)
    cnn_logits = cnn(mel_tensor, return_embed=False)
    cnn_embed = cnn(mel_tensor, return_embed=True)
    print(f" [+] Stream B Logits shape  : {cnn_logits.shape} (Expected: ({batch_size}, 2))")
    print(f" [+] Stream B Embed shape   : {cnn_embed.shape} (Expected: ({batch_size}, 128))")
    assert cnn_logits.shape == (batch_size, 2)
    assert cnn_embed.shape == (batch_size, 128)
    print(" [OK] Stream B (4-Block CNN) PASSED.")
    
    # 3. Dual-Stream Proposed Network
    model = DualStreamDetector(lfcc_dim=120, mel_channels=1, num_classes=2)
    print("\n" + model.summary() + "\n")
    
    total_params, _ = model.count_parameters()
    assert total_params < 500000, f"Parameter count {total_params} exceeds 0.5M budget!"
    
    logits = model(lfcc_tensor, mel_tensor)
    probs = model.predict_proba(lfcc_tensor, mel_tensor)
    print(f" [+] Dual-Stream Logits     : {logits.shape} (Expected: ({batch_size}, 2))")
    print(f" [+] Posterior Probs P(Y=1) : {probs.shape} (Expected: ({batch_size},))")
    print(f" [+] Sample P(Y=1|x) values : {probs.detach().numpy().round(4)}")
    assert logits.shape == (batch_size, 2)
    assert probs.shape == (batch_size,)
    print(" [OK] Dual-Stream Network PASSED.")


def test_gradient_flow_and_training_step():
    print("\n" + "=" * 60)
    print(" [TEST 3] Gradient Flow & Optimization Step Verification")
    print("=" * 60)
    
    batch_size = 4
    lfcc_tensor = torch.randn(batch_size, 120, requires_grad=True)
    mel_tensor = torch.randn(batch_size, 1, 128, 126, requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    
    model = DualStreamDetector()
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    logits = model(lfcc_tensor, mel_tensor)
    loss = criterion(logits, labels)
    loss.backward()
    
    # Check that gradients are computed for all components
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed!"
    assert all(not torch.isnan(torch.tensor(g)) for g in grad_norms), "NaN gradients detected!"
    
    optimizer.step()
    print(f" [+] Loss value             : {loss.item():.4f}")
    print(f" [+] Average gradient norm  : {sum(grad_norms)/len(grad_norms):.6f}")
    print(" [OK] Gradient flow & backpropagation PASSED.")


def test_dataset_pipeline():
    print("\n" + "=" * 60)
    print(" [TEST 4] Dataset & Batch Loader Pipeline Verification")
    print("=" * 60)
    
    feat_extractor = DualStreamFeatureExtractor()
    dataset = SyntheticDemoDataset(num_samples=16, max_len=64000)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    
    for batch_idx, batch in enumerate(loader):
        wav = batch["waveform"]
        label = batch["label"]
        audio_ids = batch["audio_id"]
        attack_ids = batch["attack_id"]
        
        lfcc, mel = feat_extractor(wav)
        assert lfcc.shape == (4, 120)
        assert mel.shape == (4, 1, 128, 126)
        assert label.shape == (4,)
        if batch_idx == 0:
            print(f" [+] Batch Waveform shape   : {wav.shape}")
            print(f" [+] Batch LFCC shape       : {lfcc.shape}")
            print(f" [+] Batch Mel shape        : {mel.shape}")
            print(f" [+] Batch Labels           : {label.tolist()}")
            print(f" [+] Batch Attack IDs       : {attack_ids}")
            
    print(" [OK] Dataset and batching pipeline PASSED.")


if __name__ == "__main__":
    print("\n============================================================")
    print("   AUDIO DEEPFAKE FORENSICS - PHASE 1-3 TEST SUITE")
    print("============================================================")
    test_feature_extractors()
    test_model_architectures()
    test_gradient_flow_and_training_step()
    test_dataset_pipeline()
    print("\n" + "=" * 60)
    print(" >>> ALL PHASE 1-3 CORE PIPELINE TESTS PASSED SUCCESSFULLY! <<<")
    print("============================================================\n")
