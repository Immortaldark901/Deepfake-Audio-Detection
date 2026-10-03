"""
Master Orchestrator for Audio Deepfake Detection Pipeline.
Phases 4-5: Training, Evaluation, and Table 3 Reproduction.

Usage:
  python run_pipeline.py --data_root /path/to/ASVspoof2019/LA \
                         --mode train_eval \
                         --model dual_stream \
                         --epochs 30 \
                         --batch_size 32 \
                         --save_dir checkpoints

  python run_pipeline.py --mode demo        # Runs full pipeline on synthetic data
  python run_pipeline.py --mode eval_only   # Evaluate from saved checkpoint
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from audio_deepfake_detector.features import DualStreamFeatureExtractor, LFCCExtractor
from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.models.baselines import (
    GMMClassifier, LogMelResNet18, LightweightRawNet2, CQCCExtractor as _CQCCExt
)
from audio_deepfake_detector.dataset import (
    ASVspoof2019LADataset, SyntheticDemoDataset, get_dataloader
)
from audio_deepfake_detector.train import (
    train_neural_model, get_class_weights,
    evaluate_neural, train_gmm_baseline, evaluate_gmm_baseline
)
from audio_deepfake_detector.evaluate import (
    compute_all_metrics, compute_per_attack_metrics,
    format_metrics_table, format_per_attack_table
)


def build_datasets_from_paths(paths, batch_size=32, num_workers=0):
    """Build ASVspoof 2019 LA train/dev/eval datasets and loaders from resolved paths dict."""
    train_ds = ASVspoof2019LADataset(
        protocol_file=paths['train_protocol'],
        audio_dir=paths['train_audio']
    )
    dev_ds = ASVspoof2019LADataset(
        protocol_file=paths['dev_protocol'],
        audio_dir=paths['dev_audio']
    )
    
    # Eval partition may not exist in all Kaggle dataset versions
    if paths.get('has_eval', False):
        eval_ds = ASVspoof2019LADataset(
            protocol_file=paths['eval_protocol'],
            audio_dir=paths['eval_audio']
        )
    else:
        print("  [INFO] Eval partition not found, using dev as eval fallback.")
        eval_ds = dev_ds
    
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dev_loader   = get_dataloader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader  = get_dataloader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, dev_ds, eval_ds, train_loader, dev_loader, eval_loader


def build_datasets_from_root(data_root, batch_size=32, num_workers=0):
    """Build datasets from a manual data_root path (legacy support)."""
    # Try to auto-detect LA directory structure
    from audio_deepfake_detector.kaggle_data import get_la_paths
    paths = get_la_paths(data_root)
    return build_datasets_from_paths(paths, batch_size, num_workers)


def build_synthetic_datasets(n_train=2000, n_val=500, n_eval=1000, batch_size=32):
    """Build synthetic datasets for demo/offline mode."""
    train_ds = SyntheticDemoDataset(n_train, max_len=64000, spoof_ratio=0.5)
    val_ds   = SyntheticDemoDataset(n_val,   max_len=64000, spoof_ratio=0.5)
    eval_ds  = SyntheticDemoDataset(n_eval,  max_len=64000, spoof_ratio=0.5)
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=batch_size, shuffle=False)
    eval_loader  = get_dataloader(eval_ds,  batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, eval_ds, train_loader, val_loader, eval_loader


def run_dual_stream(device, train_loader, val_loader, eval_loader, train_ds, epochs, save_dir):
    """Train and evaluate the proposed Dual-Stream model."""
    print("\n" + "="*60)
    print("  PROPOSED: Dual-Stream LFCC-MLP + 4-Block CNN")
    print("="*60)

    feat_ext = DualStreamFeatureExtractor().to(device)
    model    = DualStreamDetector().to(device)

    # Compute class weights from training set
    train_labels = [s['label'] for s in train_ds.samples]
    weights = get_class_weights(train_labels, device=device)
    print(f"  Class weights: bonafide={weights[0]:.3f}, spoof={weights[1]:.3f}")

    history = train_neural_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        feature_extractor=feat_ext,
        device=device,
        epochs=epochs,
        lr=1e-4,
        weight_decay=1e-4,
        class_weights=weights,
        save_dir=save_dir,
        model_name='dual_stream'
    )

    # Load best checkpoint and evaluate on eval set
    ckpt = torch.load(os.path.join(save_dir, 'dual_stream_best.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    dev_threshold = ckpt.get('eer_threshold', ckpt.get('val_metrics', {}).get('eer_threshold', 0.5))
    print(f"  Using frozen DEV threshold tau* = {dev_threshold:.5f} for EVAL partition")

    eval_metrics = evaluate_neural(model, eval_loader, feat_ext, device, threshold=dev_threshold)
    print("\n" + format_metrics_table(eval_metrics, "Proposed Dual-Stream  [EVAL]"))

    # Per-attack breakdown (A07-A19)
    per_atk = compute_per_attack_metrics(
        eval_metrics['labels'],
        eval_metrics['scores'],
        eval_metrics['attack_ids']
    )
    if per_atk:
        print("\n" + format_per_attack_table(per_atk, "Per-Attack Breakdown (A07-A19)"))

    return eval_metrics, history


def run_resnet18_baseline(device, train_loader, val_loader, eval_loader, train_ds, epochs, save_dir):
    """M3: Log-Mel ResNet-18 baseline."""
    print("\n" + "="*60)
    print("  M3: Log-Mel ResNet-18 [Frank et al. 2021]")
    print("="*60)

    feat_ext = DualStreamFeatureExtractor().to(device)
    model    = LogMelResNet18().to(device)
    print(f"  ResNet-18 parameters: {model.count_parameters():,}")

    train_labels = [s['label'] for s in train_ds.samples]
    weights = get_class_weights(train_labels, device=device)

    # Wrapper: ResNet-18 only takes mel
    class MelOnlyWrapper(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, lfcc, mel): return self.base(mel)

    wrapped = MelOnlyWrapper(model).to(device)

    history = train_neural_model(
        model=wrapped,
        train_loader=train_loader,
        val_loader=val_loader,
        feature_extractor=feat_ext,
        device=device,
        epochs=epochs,
        lr=1e-4,
        weight_decay=1e-4,
        class_weights=weights,
        save_dir=save_dir,
        model_name='m3_resnet18'
    )

    ckpt = torch.load(os.path.join(save_dir, 'm3_resnet18_best.pth'), map_location=device, weights_only=False)
    wrapped.load_state_dict(ckpt['model_state'])
    dev_threshold = ckpt.get('eer_threshold', ckpt.get('val_metrics', {}).get('eer_threshold', 0.5))
    print(f"  Using frozen DEV threshold tau* = {dev_threshold:.5f} for EVAL partition")

    eval_metrics = evaluate_neural(wrapped, eval_loader, feat_ext, device, threshold=dev_threshold)
    print("\n" + format_metrics_table(eval_metrics, "M3 Log-Mel ResNet-18 [EVAL]"))
    return eval_metrics, history


def run_rawnet2_baseline(device, train_loader, val_loader, eval_loader, train_ds, epochs, save_dir):
    """M4: Lightweight RawNet2 baseline."""
    print("\n" + "="*60)
    print("  M4: Lightweight RawNet2 [Tak et al. 2021]")
    print("="*60)

    model = LightweightRawNet2().to(device)
    print(f"  RawNet2 parameters: {model.count_parameters():,}")

    train_labels = [s['label'] for s in train_ds.samples]
    weights = get_class_weights(train_labels, device=device)

    history = train_neural_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        feature_extractor=None,   # end-to-end: raw waveform input
        device=device,
        epochs=epochs,
        lr=1e-4,
        weight_decay=1e-4,
        class_weights=weights,
        save_dir=save_dir,
        model_name='m4_rawnet2'
    )

    ckpt = torch.load(os.path.join(save_dir, 'm4_rawnet2_best.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    dev_threshold = ckpt.get('eer_threshold', ckpt.get('val_metrics', {}).get('eer_threshold', 0.5))
    print(f"  Using frozen DEV threshold tau* = {dev_threshold:.5f} for EVAL partition")

    eval_metrics = evaluate_neural(model, eval_loader, None, device, threshold=dev_threshold)
    print("\n" + format_metrics_table(eval_metrics, "M4 Lightweight RawNet2 [EVAL]"))
    return eval_metrics, history


def print_comparison_table(all_results):
    """
    Prints Table 3 from the paper: comparative performance across all models.
    """
    print("\n" + "="*85)
    print("  TABLE 3: Comparative Performance (ASVspoof 2019 LA Evaluation Partition)")
    print("="*85)
    header = f"  {'Model':<30} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'AUC%':>7} {'EER%':>7}"
    print(header)
    print("-"*85)
    for name, metrics in all_results.items():
        if metrics:
            print(
                f"  {name:<30}"
                f" {metrics.get('accuracy',0)*100:>7.2f}"
                f" {metrics.get('precision',0)*100:>7.2f}"
                f" {metrics.get('recall',0)*100:>7.2f}"
                f" {metrics.get('f1',0)*100:>7.2f}"
                f" {metrics.get('roc_auc',0)*100:>7.2f}"
                f" {metrics.get('eer',0)*100:>7.2f}"
            )
    print("="*85)


def save_results(all_results, save_dir):
    """Saves all metric results as JSON for reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'results_table3.json')
    serializable = {}
    for name, m in all_results.items():
        if m:
            serializable[name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in m.items()
                if k not in ('labels', 'scores', 'attack_ids')
            }
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Audio Deepfake Detection Pipeline")
    parser.add_argument('--data_root',  type=str, default=None,
                        help='Path to ASVspoof2019/LA directory (manual)')
    parser.add_argument('--kaggle',     action='store_true',
                        help='Auto-download ASVspoof 2019 via kagglehub')
    parser.add_argument('--mode',       type=str, default='demo',
                        choices=['demo', 'train_eval', 'eval_only'],
                        help='Execution mode')
    parser.add_argument('--models',     type=str, nargs='+',
                        default=['dual_stream', 'm3_resnet18', 'm4_rawnet2'],
                        choices=['dual_stream', 'm3_resnet18', 'm4_rawnet2'],
                        help='Models to train/evaluate')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir',   type=str, default='checkpoints')
    parser.add_argument('--device',     type=str, default='auto')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n  Running on: {device}")

    # Dataset loading
    if args.kaggle:
        print("\n  [KAGGLE] Downloading ASVspoof 2019 dataset via kagglehub...")
        from audio_deepfake_detector.kaggle_data import download_asvspoof2019, get_la_paths, quick_dataset_summary
        data_root = download_asvspoof2019()
        paths = get_la_paths(data_root)
        quick_dataset_summary(paths)
        train_ds, val_ds, eval_ds, train_loader, val_loader, eval_loader = \
            build_datasets_from_paths(paths, args.batch_size)
    elif args.data_root is not None:
        print(f"\n  [DATASET] Loading ASVspoof 2019 LA from: {args.data_root}")
        train_ds, val_ds, eval_ds, train_loader, val_loader, eval_loader = \
            build_datasets_from_root(args.data_root, args.batch_size)
    else:
        print("\n  [DEMO MODE] Using synthetic data (no ASVspoof2019 dataset required)")
        train_ds, val_ds, eval_ds, train_loader, val_loader, eval_loader = \
            build_synthetic_datasets(
                n_train=args.batch_size * 20,
                n_val=args.batch_size * 8,
                n_eval=args.batch_size * 15,
                batch_size=args.batch_size
            )

    # ── Run models ──
    all_results = {}

    if 'dual_stream' in args.models:
        eval_m, _ = run_dual_stream(device, train_loader, val_loader, eval_loader, train_ds, args.epochs, args.save_dir)
        all_results['Proposed Dual-Stream'] = eval_m

    if 'm3_resnet18' in args.models:
        eval_m, _ = run_resnet18_baseline(device, train_loader, val_loader, eval_loader, train_ds, args.epochs, args.save_dir)
        all_results['M3 Log-Mel ResNet-18'] = eval_m

    if 'm4_rawnet2' in args.models:
        eval_m, _ = run_rawnet2_baseline(device, train_loader, val_loader, eval_loader, train_ds, args.epochs, args.save_dir)
        all_results['M4 Lightweight RawNet2'] = eval_m

    # ── Final comparison table ──
    if all_results:
        print_comparison_table(all_results)
        save_results(all_results, args.save_dir)


if __name__ == '__main__':
    main()
