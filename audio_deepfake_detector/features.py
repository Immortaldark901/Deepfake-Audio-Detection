"""
Feature Extraction Modules for Audio Deepfake Forensics:
- Stream A: Linear Frequency Cepstral Coefficients (LFCC) + Delta + Delta-Delta (120-dim feature vector)
- Stream B: Log-Mel Spectrogram (128 Mel bins, 1 x 128 x 126 representation)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_linear_filterbanks(n_filters=20, n_fft=512, sample_rate=16000, low_freq=0, high_freq=8000):
    """
    Constructs a linear triangular filterbank.
    
    Args:
        n_filters (int): Number of linear filterbanks (default: 20).
        n_fft (int): FFT size (default: 512).
        sample_rate (int): Sampling rate in Hz (default: 16000).
        low_freq (float): Minimum frequency in Hz (default: 0).
        high_freq (float): Maximum frequency in Hz (default: 8000).
        
    Returns:
        torch.Tensor: Filterbank matrix of shape (n_filters, n_fft // 2 + 1).
    """
    num_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, num_bins)
    filter_freqs = torch.linspace(low_freq, high_freq, n_filters + 2)
    
    fbank = torch.zeros(n_filters, num_bins)
    for i in range(n_filters):
        f_left = filter_freqs[i]
        f_center = filter_freqs[i + 1]
        f_right = filter_freqs[i + 2]
        
        # Left slope
        left_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        if f_center > f_left:
            fbank[i, left_mask] = (fft_freqs[left_mask] - f_left) / (f_center - f_left)
            
        # Right slope
        right_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        if f_right > f_center:
            fbank[i, right_mask] = (f_right - fft_freqs[right_mask]) / (f_right - f_center)
            
    return fbank


def get_mel_filterbanks(n_mels=128, n_fft=1024, sample_rate=16000, low_freq=0, high_freq=8000):
    """
    Constructs a Mel-scale triangular filterbank.
    """
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    num_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, num_bins)
    
    mel_min = hz_to_mel(low_freq)
    mel_max = hz_to_mel(high_freq)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    fbank = np.zeros((n_mels, num_bins), dtype=np.float32)
    for i in range(n_mels):
        f_left = hz_points[i]
        f_center = hz_points[i + 1]
        f_right = hz_points[i + 2]
        
        left_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        if f_center > f_left:
            fbank[i, left_mask] = (fft_freqs[left_mask] - f_left) / (f_center - f_left)
            
        right_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        if f_right > f_center:
            fbank[i, right_mask] = (f_right - fft_freqs[right_mask]) / (f_right - f_center)
            
    return torch.from_numpy(fbank)


def dct_matrix(n_filters=20, n_ceps=20):
    """
    Computes Discrete Cosine Transform (DCT-II) orthogonal matrix.
    """
    matrix = torch.zeros(n_ceps, n_filters)
    for k in range(n_ceps):
        for n in range(n_filters):
            factor = math.sqrt(1.0 / n_filters) if k == 0 else math.sqrt(2.0 / n_filters)
            matrix[k, n] = factor * math.cos(math.pi * k * (2 * n + 1) / (2.0 * n_filters))
    return matrix


def compute_deltas(cepstra, order=2):
    """
    Computes delta / delta-delta coefficients along time axis.
    
    Args:
        cepstra (torch.Tensor): Shape (Batch, Channels, Time)
        order (int): Delta window parameter (default: 2)
        
    Returns:
        torch.Tensor: Delta coefficients with same shape.
    """
    denom = 2 * sum(n ** 2 for n in range(1, order + 1))
    pad = order
    padded = F.pad(cepstra, (pad, pad), mode='replicate')
    
    delta = torch.zeros_like(cepstra)
    for n in range(1, order + 1):
        delta = delta + n * (padded[:, :, (pad + n):(pad + n + cepstra.shape[-1])] -
                             padded[:, :, (pad - n):(pad - n + cepstra.shape[-1])])
    return delta / denom


class LFCCExtractor(nn.Module):
    """
    Linear Frequency Cepstral Coefficients (LFCC) Extractor (Stream A).
    Extracts Static (20), Delta (20), and Delta-Delta (20) features,
    pooled to create a compact, highly discriminative 120-dim feature vector (mean & std).
    """
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_filters=20, n_ceps=20):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_filters = n_filters
        self.n_ceps = n_ceps
        
        # Register window, filterbank, and DCT matrix as buffers
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        
        fbank = get_linear_filterbanks(n_filters, n_fft, sample_rate)
        self.register_buffer("linear_fbank", fbank)
        
        dct_mat = dct_matrix(n_filters, n_ceps)
        self.register_buffer("dct_matrix", dct_mat)

    def forward(self, x, return_frames=False):
        """
        Args:
            x (torch.Tensor): Raw audio waveform of shape (Batch, Time) or (Time,)
            return_frames (bool): If True, returns full 60xT frame-level features.
            
        Returns:
            torch.Tensor: 120-dim feature vector of shape (Batch, 120) by default.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        
        # 1. STFT Spectrogram
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # 2. Power Spectrum: |STFT|^2 -> shape (Batch, n_fft // 2 + 1, Frames)
        power_spec = torch.abs(stft) ** 2
        
        # 3. Linear Filterbank Energy: (n_filters, num_bins) x (Batch, num_bins, Frames)
        # -> shape (Batch, n_filters, Frames)
        fbank_energy = torch.matmul(self.linear_fbank, power_spec)
        
        # 4. Log Energy
        log_energy = torch.log(torch.clamp(fbank_energy, min=1e-12))
        
        # 5. Discrete Cosine Transform (DCT-II): (n_ceps, n_filters) x (Batch, n_filters, Frames)
        # -> shape (Batch, n_ceps, Frames)
        static_lfcc = torch.matmul(self.dct_matrix, log_energy)
        
        # 6. Delta and Delta-Delta calculation
        delta_lfcc = compute_deltas(static_lfcc, order=2)
        ddelta_lfcc = compute_deltas(delta_lfcc, order=2)
        
        # Combined frame-level LFCC: shape (Batch, 60, Frames)
        frame_features = torch.cat([static_lfcc, delta_lfcc, ddelta_lfcc], dim=1)
        
        if return_frames:
            return frame_features
            
        # 7. Statistical Pooling (Mean & Std over time axis) -> 60*2 = 120 dimensional vector
        feat_mean = torch.mean(frame_features, dim=-1)   # (Batch, 60)
        feat_std = torch.std(frame_features, dim=-1)     # (Batch, 60)
        
        lfcc_120 = torch.cat([feat_mean, feat_std], dim=-1)  # (Batch, 120)
        return lfcc_120


class LogMelExtractor(nn.Module):
    """
    Log-Mel Spectrogram Extractor (Stream B).
    Computes 128-bin Mel-spectrogram with psychoacoustic compression and fixed
    time-frequency grid representation of (1, 128, 126) for the 4-block CNN.
    """
    def __init__(self, sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128, target_frames=126):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_frames = target_frames
        
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        
        mel_fbank = get_mel_filterbanks(n_mels, n_fft, sample_rate)
        self.register_buffer("mel_fbank", mel_fbank)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw audio waveform of shape (Batch, Time) or (Time,)
            
        Returns:
            torch.Tensor: Log-Mel spectrogram of shape (Batch, 1, 128, target_frames)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 1. STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # 2. Power Spectrum: (Batch, n_fft // 2 + 1, Frames)
        power_spec = torch.abs(stft) ** 2
        
        # 3. Mel Filterbank Energy: (Batch, n_mels, Frames)
        mel_energy = torch.matmul(self.mel_fbank, power_spec)
        
        # 4. Logarithmic Scaling: log(1 + x)
        log_mel = torch.log1p(mel_energy)
        
        # 5. Normalization per utterance
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_mel = (log_mel - mean) / std
        
        # Add channel dimension: (Batch, 1, n_mels, Frames)
        norm_mel = norm_mel.unsqueeze(1)
        
        # 6. Adaptive resize/pooling to fixed target frames (128 x target_frames)
        if norm_mel.shape[-1] != self.target_frames:
            norm_mel = F.adaptive_avg_pool2d(norm_mel, (self.n_mels, self.target_frames))
            
        return norm_mel


class DualStreamFeatureExtractor(nn.Module):
    """
    Combined Front-end Feature Extraction Module providing synchronous
    extraction of both Stream A (LFCC) and Stream B (Log-Mel) from raw audio.
    """
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.stream_a_lfcc = LFCCExtractor(sample_rate=sample_rate)
        self.stream_b_mel = LogMelExtractor(sample_rate=sample_rate)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw waveform tensor of shape (Batch, Time)
            
        Returns:
            tuple: (lfcc_120, log_mel_spec)
                - lfcc_120: shape (Batch, 120)
                - log_mel_spec: shape (Batch, 1, 128, 126)
        """
        lfcc_120 = self.stream_a_lfcc(x)
        log_mel_spec = self.stream_b_mel(x)
        return lfcc_120, log_mel_spec
