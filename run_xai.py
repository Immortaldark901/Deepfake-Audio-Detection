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
from audio_deepfake_detector.xai import DualStreamGradCAM, compute_frequency_marginals, evaluate_perturbation_masking, plot_gradcam_explanation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=None)
    p.add_argument('--kaggle', action='store_true')
    p.add_argument('--checkpoint', default='results_route1/proposed_dual_stream_best.pth')
    p.add_argument('--output_dir', default='results_route1/xai')
    p.add_argument('--max_samples', type=int, default=100)
    p.add_argument('--top_k_bands', type=int, default=30)
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
    feat = DualStreamFeatureExtractor().to(device)
    cam_extractor = DualStreamGradCAM(model)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    count = 0
    for item in ds.samples:
        if item['label'] != 1:
            continue
        wav = load_audio_file(item['path']).to(device)
        with torch.no_grad():
            lfcc, mel = feat(wav.unsqueeze(0))
            prob = torch.softmax(model(lfcc, mel), dim=-1)[0, 1].item()
        if prob < 0.5:
            continue

        cam, _, s_orig = cam_extractor.generate_cam(lfcc, mel, target_class=1)
        marginals, freqs, high_ratio = compute_frequency_marginals(cam)
        delta, s0, sm, masked = evaluate_perturbation_masking(model, lfcc, mel, cam, top_k_bands=args.top_k_bands, target_class=1)

        rows.append({
            'audio_id': item['audio_id'],
            'attack_id': item['attack_id'],
            'original_spoof_probability': float(s0),
            'masked_spoof_probability': float(sm),
            'relative_confidence_drop_percent': float(delta),
            'attribution_ratio_above_4khz': float(high_ratio)
        })

        if count < 3:
            plot_gradcam_explanation(
                mel.squeeze().detach().cpu().numpy(), cam, marginals, freqs,
                title=f"Grad-CAM | {item['attack_id']} | {item['audio_id']}",
                save_path=os.path.join(args.output_dir, f"example_{count+1}_{item['attack_id']}.png")
            )
        count += 1
        if count >= args.max_samples:
            break

    if not rows:
        raise RuntimeError('No correctly detected spoof samples were available for XAI.')

    drops = np.array([r['relative_confidence_drop_percent'] for r in rows])
    ratios = np.array([r['attribution_ratio_above_4khz'] for r in rows])
    summary = {
        'n_samples': len(rows),
        'mean_relative_confidence_drop_percent': float(drops.mean()),
        'median_relative_confidence_drop_percent': float(np.median(drops)),
        'std_relative_confidence_drop_percent': float(drops.std()),
        'mean_attribution_ratio_above_4khz': float(ratios.mean()),
        'top_k_bands_masked': args.top_k_bands
    }
    with open(os.path.join(args.output_dir, 'xai_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, 'xai_samples.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
