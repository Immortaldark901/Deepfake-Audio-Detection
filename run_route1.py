import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from audio_deepfake_detector.dataset import ASVspoof2019LADataset, get_dataloader
from audio_deepfake_detector.kaggle_data import download_asvspoof2019, get_la_paths, quick_dataset_summary
from audio_deepfake_detector.features import DualStreamFeatureExtractor
from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.models.baselines import LogMelResNet18
from audio_deepfake_detector.train import train_neural_model, evaluate_neural, get_class_weights
from audio_deepfake_detector.evaluate import compute_per_attack_metrics, format_metrics_table


class MelOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, lfcc, mel):
        return self.model(mel)


def build_datasets(data_root, batch_size, num_workers):
    paths = get_la_paths(data_root)
    quick_dataset_summary(paths)
    if not paths['has_eval']:
        raise RuntimeError('ASVspoof 2019 LA evaluation audio/protocol was not found. Do not use dev as eval for the final paper.')

    train_ds = ASVspoof2019LADataset(paths['train_protocol'], paths['train_audio'])
    dev_ds = ASVspoof2019LADataset(paths['dev_protocol'], paths['dev_audio'])
    eval_ds = ASVspoof2019LADataset(paths['eval_protocol'], paths['eval_audio'])

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = get_dataloader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = get_dataloader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, dev_ds, eval_ds, train_loader, dev_loader, eval_loader


def train_and_eval(model, model_name, train_ds, train_loader, dev_loader, eval_loader, device, epochs, save_dir, feature_extractor):
    labels = [s['label'] for s in train_ds.samples]
    weights = get_class_weights(labels, device=device)
    print('Class weights:', weights.detach().cpu().numpy())

    history = train_neural_model(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        feature_extractor=feature_extractor,
        device=device,
        epochs=epochs,
        lr=1e-4,
        weight_decay=1e-4,
        class_weights=weights,
        save_dir=save_dir,
        model_name=model_name
    )

    ckpt = torch.load(os.path.join(save_dir, model_name + '_best.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    dev_threshold = ckpt.get('eer_threshold', ckpt.get('val_metrics', {}).get('eer_threshold', 0.5))
    print(f"Using frozen DEV decision threshold tau* = {dev_threshold:.5f} for EVAL evaluation.")
    metrics = evaluate_neural(model, eval_loader, feature_extractor, device, threshold=dev_threshold)
    print(format_metrics_table(metrics, model_name + ' | ASVspoof 2019 LA EVAL'))

    per_attack = compute_per_attack_metrics(metrics['labels'], metrics['scores'], metrics['attack_ids'])
    with open(os.path.join(save_dir, model_name + '_per_attack.json'), 'w') as f:
        json.dump(per_attack, f, indent=2)

    serializable = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items() if k not in ['labels', 'scores', 'attack_ids']}
    with open(os.path.join(save_dir, model_name + '_metrics.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    np.savez(os.path.join(save_dir, model_name + '_scores.npz'), labels=metrics['labels'], scores=metrics['scores'], attack_ids=metrics['attack_ids'])
    return model, metrics, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--models', nargs='+', choices=['dual_stream', 'resnet18'], default=['dual_stream', 'resnet18'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', default='results_route1')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if not args.kaggle and args.data_root is None:
        raise ValueError('Use --kaggle or --data_root PATH')
    data_root = download_asvspoof2019() if args.kaggle else args.data_root

    os.makedirs(args.save_dir, exist_ok=True)
    train_ds, dev_ds, eval_ds, train_loader, dev_loader, eval_loader = build_datasets(data_root, args.batch_size, args.num_workers)
    feature_extractor = DualStreamFeatureExtractor().to(device)
    feature_extractor.eval()

    results = {}
    if 'dual_stream' in args.models:
        model = DualStreamDetector().to(device)
        print(model.summary())
        model, metrics, _ = train_and_eval(model, 'proposed_dual_stream', train_ds, train_loader, dev_loader, eval_loader, device, args.epochs, args.save_dir, feature_extractor)
        results['Proposed Dual-Stream'] = {k: float(v) for k, v in metrics.items() if k not in ['labels', 'scores', 'attack_ids']}

    if 'resnet18' in args.models:
        base = LogMelResNet18().to(device)
        model = MelOnlyWrapper(base).to(device)
        print('ResNet-18 parameters:', base.count_parameters())
        model, metrics, _ = train_and_eval(model, 'm3_logmel_resnet18', train_ds, train_loader, dev_loader, eval_loader, device, args.epochs, args.save_dir, feature_extractor)
        results['M3 Log-Mel ResNet-18'] = {k: float(v) for k, v in metrics.items() if k not in ['labels', 'scores', 'attack_ids']}

    with open(os.path.join(args.save_dir, 'neural_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
