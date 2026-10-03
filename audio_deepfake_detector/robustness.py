"""
Phase 7: Acoustic Stress Testing & Continuous AWGN SNR Sweep.
Injects calibrated Additive White Gaussian Noise (AWGN) at precise SNR levels
[Clean, 30 dB, 20 dB, 10 dB, 0 dB] and profiles degradation of EER, AUC, and Accuracy.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .evaluate import compute_all_metrics


def inject_awgn(waveform, snr_db, clip=True, return_snr=False):
    """
    Injects calibrated Additive White Gaussian Noise (AWGN) into audio waveform.
    
    Formula:
        sigma_noise^2 = sigma_signal^2 / 10^(SNR / 10)
        w[n] ~ N(0, sigma_noise^2)
        y[n] = x[n] + w[n]
        
    Args:
        waveform (np.ndarray or torch.Tensor): 1D audio waveform (Samples,)
        snr_db (float or None): Signal-to-Noise Ratio in dB. If None or np.inf, returns clean.
        clip (bool): Whether to clip output to [-1.0, 1.0] range.
        return_snr (bool): If True, returns (noisy_waveform, achieved_snr_db).
        
    Returns:
        noisy_waveform or (noisy_waveform, achieved_snr_db)
    """
    if snr_db is None or np.isinf(snr_db):
        if return_snr:
            return waveform, float('inf')
        return waveform
        
    is_torch = isinstance(waveform, torch.Tensor)
    if is_torch:
        x = waveform.cpu().numpy()
    else:
        x = np.copy(waveform)
        
    # Calculate signal power
    sig_power = float(np.mean(x ** 2))
    if sig_power <= 1e-12:
        if return_snr:
            return waveform, float('inf')
        return waveform
        
    # Calculate required noise power for target SNR
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, size=x.shape).astype(x.dtype)
    noisy_x = x + noise
    
    # Calculate actual achieved SNR before clipping
    actual_noise_power = float(np.mean(noise ** 2))
    if actual_noise_power > 1e-12:
        achieved_snr = 10.0 * np.log10(sig_power / actual_noise_power)
    else:
        achieved_snr = float('inf')
    
    # Clip to standard [-1.0, 1.0] audio range if requested
    if clip:
        noisy_x = np.clip(noisy_x, -1.0, 1.0)
    
    if is_torch:
        out_wav = torch.from_numpy(noisy_x).to(waveform.device)
    else:
        out_wav = noisy_x
        
    if return_snr:
        return out_wav, float(achieved_snr)
    return out_wav


def run_robustness_sweep(model, dataset, feature_extractor, snr_levels=[np.inf, 30, 20, 10, 0],
                         device='cpu', batch_size=32, max_samples=None, threshold=0.5):
    """
    Evaluates a trained model across a sweep of calibrated SNR conditions.
    
    Args:
        model: Trained detector (DualStreamDetector or baseline)
        dataset: Dataset providing raw audio waveforms and labels
        feature_extractor: DualStreamFeatureExtractor
        snr_levels: List of SNRs in dB to test (e.g. [inf, 30, 20, 10, 0])
        device: 'cuda' or 'cpu'
        batch_size: Evaluation batch size
        max_samples: Optional cap on test samples for rapid benchmarking
        threshold: Frozen decision threshold tau* (from DEV)
        
    Returns:
        results: Dict mapping snr -> metrics dict (eer, roc_auc, accuracy, f1, achieved_snr_db)
    """
    model.eval()
    results = {}
    
    num_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    
    for snr in snr_levels:
        snr_label = "Clean" if np.isinf(snr) else f"{int(snr)}dB"
        y_true = []
        y_scores = []
        measured_snrs = []
        
        for idx in range(num_samples):
            item = dataset[idx]
            waveform = item['waveform'] if 'waveform' in item else item.get('audio')
            label = item['label']
            
            # Inject AWGN at target SNR and measure actual SNR
            noisy_audio, actual_snr = inject_awgn(waveform, snr, return_snr=True)
            if not np.isinf(actual_snr):
                measured_snrs.append(actual_snr)
            
            # Convert to torch tensor if needed
            if not isinstance(noisy_audio, torch.Tensor):
                noisy_tensor = torch.from_numpy(noisy_audio).float()
            else:
                noisy_tensor = noisy_audio.float()
                
            if noisy_tensor.ndim == 1:
                noisy_tensor = noisy_tensor.unsqueeze(0)  # (1, Samples)
                
            # Extract features on noisy audio (Batch, 120) and (Batch, 1, 128, 126)
            with torch.no_grad():
                lfcc, mel = feature_extractor(noisy_tensor.to(device))
                logits = model(lfcc, mel)
                probs = torch.softmax(logits, dim=-1)
                spoof_prob = probs[0, 1].item()
                
            y_true.append(label)
            y_scores.append(spoof_prob)
            
        metrics = compute_all_metrics(np.array(y_true), np.array(y_scores), threshold=threshold)
        metrics['target_snr_db'] = None if np.isinf(snr) else float(snr)
        metrics['measured_snr_db'] = float(np.mean(measured_snrs)) if measured_snrs else None
        results[snr_label] = metrics
    return results


def plot_robustness_curves(sweep_results, title="Acoustic Stress Testing (AWGN SNR Sweep)", save_path=None):
    """
    Generates a 2-panel figure showing EER and ROC-AUC degradation across SNR levels.
    """
    snr_keys = list(sweep_results.keys())
    eers = [sweep_results[k]['eer'] for k in snr_keys]
    aucs = [sweep_results[k]['roc_auc'] for k in snr_keys]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # 1. EER Curve
    ax1.plot(snr_keys, eers, marker='o', color='#e74c3c', linewidth=2.5, markersize=8)
    ax1.set_title("Equal Error Rate (EER %) vs SNR", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("EER (%) — Lower is Better")
    ax1.grid(True, linestyle='--', alpha=0.5)
    for i, txt in enumerate(eers):
        ax1.annotate(f"{txt:.2f}%", (snr_keys[i], eers[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
        
    # 2. ROC-AUC Curve
    ax2.plot(snr_keys, aucs, marker='s', color='#2980b9', linewidth=2.5, markersize=8)
    ax2.set_title("ROC-AUC vs SNR", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_ylabel("ROC-AUC — Higher is Better")
    ax2.set_ylim([0.45, 1.02])
    ax2.grid(True, linestyle='--', alpha=0.5)
    for i, txt in enumerate(aucs):
        ax2.annotate(f"{txt:.3f}", (snr_keys[i], aucs[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
        
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    return fig
