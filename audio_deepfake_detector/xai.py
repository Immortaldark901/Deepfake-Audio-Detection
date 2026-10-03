"""
Explainable AI (XAI) & Attribution Pipeline using Grad-CAM.
Extracts time-frequency saliency maps from Stream B (4-Block CNN) of the Dual-Stream network.
Performs frequency-band attribution analysis and perturbation masking.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DualStreamGradCAM:
    """
    Grad-CAM extractor tailored for DualStreamDetector (Stream B 4-Block CNN).
    
    Computes class-discriminative localization map L^c_{Grad-CAM} by backpropagating
    the score for target class c into the feature maps of the 4th convolutional block.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam(self, lfcc, mel, target_class=1):
        """
        Generate Grad-CAM heatmap for the given inputs.
        
        Args:
            lfcc (torch.Tensor): LFCC features (1, 120) or (Batch, 120)
            mel (torch.Tensor): Log-Mel spectrogram (1, 1, 128, 126)
            target_class (int): Target class index (1 = Spoof, 0 = Bonafide)
            
        Returns:
            cam (np.ndarray): 2D heatmap resized to match mel shape (128, 126), normalized [0, 1]
            logits (torch.Tensor): Raw classification logits
            pred_prob (float): Posterior probability for target_class
        """
        if lfcc.ndim == 1:
            lfcc = lfcc.unsqueeze(0)
        if mel.ndim == 3:
            mel = mel.unsqueeze(0)
            
        lfcc = lfcc.clone().detach().requires_grad_(False)
        mel = mel.clone().detach().requires_grad_(True)
        
        # Forward pass with cam retention
        self.model.zero_grad()
        logits = self.model(lfcc, mel, retain_cam_maps=True)
        
        probs = F.softmax(logits, dim=-1)
        pred_prob = probs[0, target_class].item()
        
        # Backward pass on target score
        score = logits[0, target_class]
        score.backward()
        
        # Retrieve feature maps and gradients from Stream B
        activations = self.model.stream_b.last_conv_act   # (1, 128, 8, 7)
        gradients = self.model.stream_b.last_conv_grad     # (1, 128, 8, 7)
        
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. Ensure retain_cam_maps=True.")
            
        # Global average pooling over gradients to compute channel weights alpha_k
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, 128, 1, 1)
        
        # Weighted combination of feature maps
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)  # (1, 1, 8, 7)
        cam = F.relu(cam)  # Only positive influence
        
        # Upsample CAM to original mel spectrogram dimensions (128, 126)
        cam_upsampled = F.interpolate(
            cam, size=(mel.shape[2], mel.shape[3]), mode='bilinear', align_corners=False
        )
        
        cam_np = cam_upsampled.squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np, logits.detach(), pred_prob


def compute_frequency_marginals(cam, sr=16000, n_mels=128, f_max=8000):
    """
    Computes frequency marginal attribution \int L(f, t) dt across mel bands.
    
    Args:
        cam (np.ndarray): Grad-CAM heatmap of shape (128, 126) [Freq, Time]
        sr (int): Sampling rate
        n_mels (int): Number of Mel frequency bins
        f_max (int): Maximum frequency
        
    Returns:
        marginals (np.ndarray): Marginal attribution per frequency bin (128,)
        mel_freqs (np.ndarray): Approximate center frequencies in Hz for each bin
        high_band_ratio (float): Ratio of attribution in upper frequency half (> 4kHz)
    """
    # Average across time dimension
    marginals = np.mean(cam, axis=1)  # Shape: (128,)
    
    # Generate approx Hz values for each mel bin
    # Slaney formula approximation
    mels = np.linspace(0, 2595 * np.log10(1 + f_max / 700), n_mels)
    mel_freqs = 700 * (10 ** (mels / 2595) - 1)
    
    # Calculate ratio of attribution in > 4kHz band (synthetic speech artifact band)
    idx_4k = np.searchsorted(mel_freqs, 4000)
    total_attribution = np.sum(marginals) + 1e-12
    high_band_attribution = np.sum(marginals[idx_4k:])
    high_band_ratio = float(high_band_attribution / total_attribution)
    
    return marginals, mel_freqs, high_band_ratio


def mask_raw_audio_frequencies(waveform, mel_band_indices, sr=16000, n_fft=1024, hop_length=160, win_length=400, n_mels=128):
    """
    Masks specified Mel frequency bands directly in the RAW AUDIO waveform using STFT/iSTFT.
    
    Args:
        waveform (torch.Tensor): 1D or 2D raw audio waveform tensor (1, T) or (T,)
        mel_band_indices (list/array): Indices of Mel bins to mask (0 to n_mels-1)
        sr (int): Sampling rate
        n_fft (int): FFT length
        hop_length (int): STFT hop length
        win_length (int): STFT window length
        n_mels (int): Total number of Mel bins
        
    Returns:
        torch.Tensor: Masked raw audio waveform of original shape and length
    """
    orig_1d = (waveform.ndim == 1)
    if orig_1d:
        wav = waveform.unsqueeze(0)
    else:
        wav = waveform
        
    length = wav.shape[-1]
    device = wav.device
    window = torch.hann_window(win_length, device=device)
    
    # Forward STFT on raw audio
    stft = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True, center=True, pad_mode='reflect'
    )
    
    # Calculate Mel band frequency edges in Hz
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_edges = mel_to_hz(mel_points)
    
    num_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, num_bins)
    
    # Find all FFT bins corresponding to selected Mel bands
    fft_mask = torch.ones(num_bins, dtype=torch.bool, device=device)
    for m_idx in mel_band_indices:
        f_low = hz_edges[m_idx]
        f_high = hz_edges[m_idx + 2]
        bin_mask = (fft_freqs >= f_low) & (fft_freqs <= f_high)
        fft_mask[torch.from_numpy(bin_mask).to(device)] = False
        
    # Apply mask across all time frames
    stft_masked = stft.clone()
    stft_masked[:, ~fft_mask, :] = 0.0 + 0.0j
    
    # Invert back to raw audio waveform via iSTFT
    masked_wav = torch.istft(
        stft_masked, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, length=length
    )
    
    if orig_1d:
        return masked_wav.squeeze(0)
    return masked_wav


def evaluate_perturbation_masking(model, lfcc, mel, cam, top_k_bands=30, target_class=1,
                                  waveform=None, feature_extractor=None):
    """
    Evaluates model confidence drop when high-attribution frequency bands are masked out
    in the RAW AUDIO WAVEFORM, followed by re-extracting BOTH LFCC and Log-Mel features.
    
    Delta S = (S(x) - S(x_masked)) / S(x) * 100%
    
    Args:
        model: Trained DualStreamDetector
        lfcc (torch.Tensor): Original LFCC features (1, 120)
        mel (torch.Tensor): Original Log-Mel spectrogram (1, 1, 128, 126)
        cam (np.ndarray): Grad-CAM heatmap (128, 126)
        top_k_bands (int): Number of most salient frequency bins to mask
        target_class (int): Class to monitor (1 = spoof)
        waveform (torch.Tensor, optional): Raw audio waveform (1, T) or (T,)
        feature_extractor (DualStreamFeatureExtractor, optional): Extractor to recompute features
        
    Returns:
        delta_s (float): Relative percentage confidence drop
        s_orig (float): Original posterior probability
        s_masked (float): Masked posterior probability
        masked_mel (torch.Tensor): Masked spectrogram tensor
    """
    model.eval()
    if lfcc.ndim == 1:
        lfcc = lfcc.unsqueeze(0)
    if mel.ndim == 3:
        mel = mel.unsqueeze(0)
        
    # Baseline prediction
    with torch.no_grad():
        logits_orig = model(lfcc, mel)
        s_orig = F.softmax(logits_orig, dim=-1)[0, target_class].item()
        
    # Identify top_k frequency bands with highest attribution
    marginals = np.mean(cam, axis=1)  # (128,)
    top_indices = np.argsort(marginals)[::-1][:top_k_bands]
    
    if waveform is not None and feature_extractor is not None:
        # 1. Mask frequency bands directly in the RAW AUDIO WAVEFORM
        masked_wav = mask_raw_audio_frequencies(waveform, top_indices)
        if masked_wav.ndim == 1:
            masked_wav = masked_wav.unsqueeze(0)
            
        # 2. Re-extract BOTH LFCC and Log-Mel from masked raw audio
        with torch.no_grad():
            masked_lfcc, masked_mel = feature_extractor(masked_wav.to(lfcc.device))
            
        # 3. Model prediction from recomputed features
        with torch.no_grad():
            logits_masked = model(masked_lfcc, masked_mel)
            s_masked = F.softmax(logits_masked, dim=-1)[0, target_class].item()
    else:
        # Fallback: tensor-level masking
        min_val = mel.min().item()
        masked_mel = mel.clone()
        for idx in top_indices:
            masked_mel[:, :, idx, :] = min_val
        with torch.no_grad():
            logits_masked = model(lfcc, masked_mel)
            s_masked = F.softmax(logits_masked, dim=-1)[0, target_class].item()
        
    # Relative drop in posterior score
    if s_orig > 1e-6:
        delta_s = ((s_orig - s_masked) / s_orig) * 100.0
    else:
        delta_s = 0.0
        
    return delta_s, s_orig, s_masked, masked_mel


def plot_gradcam_explanation(mel_np, cam, marginals, mel_freqs, title="Grad-CAM XAI Attribution", save_path=None):
    """
    Creates a publication-grade 3-panel figure:
    1. Log-Mel Spectrogram
    2. Grad-CAM Heatmap Overlay
    3. Frequency Marginal Profile (showing high-frequency attribution)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={'width_ratios': [1, 1, 0.6]})
    
    # 1. Original Spectrogram
    im1 = ax1.imshow(mel_np, origin='lower', aspect='auto', cmap='magma')
    ax1.set_title("Log-Mel Spectrogram (128×126)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Time Frames")
    ax1.set_ylabel("Mel Frequency Bins")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Grad-CAM Overlay
    ax2.imshow(mel_np, origin='lower', aspect='auto', cmap='gray', alpha=0.5)
    im2 = ax2.imshow(cam, origin='lower', aspect='auto', cmap='jet', alpha=0.6)
    ax2.set_title("Grad-CAM Attribution Overlay", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time Frames")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Frequency Marginals
    ax3.plot(marginals, np.arange(len(marginals)), color='#e74c3c', linewidth=2)
    ax3.set_title("Frequency Attribution ∫L dt", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Mean Saliency")
    ax3.set_ylabel("Mel Bins (0 to 8 kHz)")
    ax3.axhspan(64, 127, color='#f39c12', alpha=0.2, label='4–8 kHz Band')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    return fig
