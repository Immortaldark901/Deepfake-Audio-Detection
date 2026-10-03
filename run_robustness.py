import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from audio_deepfake_detector.dataset import ASVspoof2019LADataset, load_audio_file
from audio_deepfake_detector.kaggle_data import get_la_paths, download_asvspoof2019
from audio_deepfake_detector.features import DualStreamFeatureExtractor
from audio_deepfake_detector.models.dual_stream import DualStreamDetector
from audio_deepfake_detector.evaluate import compute_all_metrics
from audio_deepfake_detector.robustness import inject_awgn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=None)
    p.add_argument('--kaggle', action='store_true')
    p.add_argument('--checkpoint', default='results_route1/proposed_dual_stream_best.pth')
    p.add_argument('--output', default='results_route1/robustness.json')
    p.add_argument('--max_samples', type=int, default=1000)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = download_asvspoof2019() if args.kaggle else args.data_root
    if data_root is None:
        raise ValueError('Use --kaggle or --data_root PATH')
    paths = get_la_paths(data_root)
    ds = ASVspoof2019LADataset(paths['eval_protocol'], paths['eval_audio'])

    model = DualStreamDetector().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    dev_threshold = ckpt.get('eer_threshold', ckpt.get('val_metrics', {}).get('eer_threshold', 0.5))
    print(f"Using frozen DEV decision threshold tau* = {dev_threshold:.5f} for robustness evaluation.")

    feat = DualStreamFeatureExtractor().to(device)
    feat.eval()
    levels = [('Clean', None), ('30dB', 30), ('20dB', 20), ('10dB', 10), ('0dB', 0)]
    results = {}

    n = min(args.max_samples, len(ds))
    for label, snr in levels:
        y_true, y_score, measured_snrs = [], [], []
        for i in range(n):
            item = ds[i]
            wav = load_audio_file(item['path'])
            noisy, actual_snr = inject_awgn(wav, snr, return_snr=True)
            if not np.isinf(actual_snr):
                measured_snrs.append(actual_snr)
            with torch.no_grad():
                lfcc, mel = feat(noisy.unsqueeze(0).to(device))
                score = torch.softmax(model(lfcc, mel), dim=-1)[0, 1].item()
            y_true.append(item['label'])
            y_score.append(score)
        metrics = compute_all_metrics(y_true, y_score, threshold=dev_threshold)
        metrics['target_snr_db'] = None if snr is None else float(snr)
        metrics['measured_snr_db'] = float(np.mean(measured_snrs)) if measured_snrs else None
        results[label] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}
        print(f"[{label}] EER={metrics['eer']*100:.2f}%, AUC={metrics['roc_auc']*100:.2f}%, Acc={metrics['accuracy']*100:.2f}%, Meas_SNR={metrics['measured_snr_db']}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
