"""
Reference Baseline Models for Table 3 Reproduction.
Paper: "Towards Trustworthy Audio Deepfake Detection for Societal Good"

Implements the four reference models:
  M1: LFCC-GMM     [Todisco et al., 2019]
  M2: CQCC-GMM     [Todisco et al., 2019]
  M3: Log-Mel ResNet-18  [Frank et al., 2021]
  M4: RawNet2 (End-to-End lightweight variant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


# ─────────────────────────────────────────────────────────────
# SHARED UTILITY: Constant-Q Cepstral Coefficient extraction
# ─────────────────────────────────────────────────────────────
class CQCCExtractor(nn.Module):
    """
    Approximate Constant-Q Cepstral Coefficients (CQCC) extractor for M2 baseline.
    Uses geometrically-spaced (log-frequency) filterbank to approximate CQT,
    followed by DCT-II cepstrum. Output: 120-dim statistical vector.
    """
    def __init__(self, sample_rate=16000, n_bins=20, n_ceps=20,
                 n_fft=512, win_length=400, hop_length=160,
                 fmin=50.0, fmax=8000.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.n_ceps = n_ceps
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        # Geometrically-spaced (log-scale) filterbank
        fbank = self._build_log_filterbank(n_bins, n_fft, sample_rate, fmin, fmax)
        self.register_buffer("cq_fbank", fbank)

        # DCT-II matrix
        dct_mat = self._dct_matrix(n_bins, n_ceps)
        self.register_buffer("dct_mat", dct_mat)

    def _build_log_filterbank(self, n_bins, n_fft, sr, fmin, fmax):
        num_bins = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sr / 2, num_bins)
        # Log-spaced center frequencies
        centers = torch.logspace(np.log10(fmin), np.log10(fmax), n_bins + 2)
        fbank = torch.zeros(n_bins, num_bins)
        for i in range(n_bins):
            fl, fc, fr = centers[i], centers[i+1], centers[i+2]
            left  = (fft_freqs >= fl) & (fft_freqs <= fc)
            right = (fft_freqs >= fc) & (fft_freqs <= fr)
            if fc > fl:
                fbank[i, left]  = (fft_freqs[left]  - fl) / (fc - fl)
            if fr > fc:
                fbank[i, right] = (fr - fft_freqs[right]) / (fr - fc)
        return fbank

    def _dct_matrix(self, n_filters, n_ceps):
        import math
        mat = torch.zeros(n_ceps, n_filters)
        for k in range(n_ceps):
            for n in range(n_filters):
                f = math.sqrt(1.0/n_filters) if k == 0 else math.sqrt(2.0/n_filters)
                mat[k, n] = f * math.cos(math.pi * k * (2*n+1) / (2.0*n_filters))
        return mat

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window,
                          return_complex=True, center=True, pad_mode='reflect')
        power = torch.abs(stft) ** 2
        energy = torch.matmul(self.cq_fbank, power)
        log_e  = torch.log(torch.clamp(energy, min=1e-12))
        cqcc   = torch.matmul(self.dct_mat, log_e)  # (B, n_ceps, T)

        # Delta + Delta-Delta
        def deltas(c, order=2):
            denom = 2 * sum(n**2 for n in range(1, order+1))
            padded = F.pad(c, (order, order), mode='replicate')
            d = torch.zeros_like(c)
            for n in range(1, order+1):
                d = d + n * (padded[:, :, order+n : order+n+c.shape[-1]] -
                              padded[:, :, order-n : order-n+c.shape[-1]])
            return d / denom

        delta  = deltas(cqcc)
        ddelta = deltas(delta)
        frames = torch.cat([cqcc, delta, ddelta], dim=1)  # (B, 60, T)
        mu  = frames.mean(dim=-1)
        std = frames.std(dim=-1)
        return torch.cat([mu, std], dim=-1)  # (B, 120)


# ─────────────────────────────────────────────────────────────
# M1 / M2: GMM Wrapper (sklearn backend, sklearn GaussianMixture)
# ─────────────────────────────────────────────────────────────
class GMMClassifier:
    """
    Binary GMM Classifier.
    Fits one GMM per class (bonafide / spoof) and computes
    log-likelihood ratio as detection score.
    Used for M1 (LFCC-GMM) and M2 (CQCC-GMM) baselines.
    """
    def __init__(self, n_components=512, covariance_type='diag', max_iter=100, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.gmm_bonafide = None
        self.gmm_spoof = None
        self.is_fitted = False

    def fit(self, X_bona, X_spoof, verbose=True):
        """
        Fit separate GMMs for bonafide and spoof classes.
        Args:
            X_bona  (np.ndarray): Features for bonafide samples (N, feat_dim)
            X_spoof (np.ndarray): Features for spoof samples   (N, feat_dim)
        """
        if verbose:
            print(f"  Fitting GMM bonafide  (n={len(X_bona)}, components={self.n_components}) ...")
        self.gmm_bonafide = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0
        )
        self.gmm_bonafide.fit(X_bona)

        if verbose:
            print(f"  Fitting GMM spoof     (n={len(X_spoof)}, components={self.n_components}) ...")
        self.gmm_spoof = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0
        )
        self.gmm_spoof.fit(X_spoof)
        self.is_fitted = True
        if verbose:
            print("  GMM fitting complete.")

    def score(self, X):
        """
        Compute log-likelihood ratio LLR = log P(X|spoof) - log P(X|bonafide).
        Higher score => more likely spoof.
        """
        assert self.is_fitted, "Call fit() before score()"
        llr = self.gmm_spoof.score_samples(X) - self.gmm_bonafide.score_samples(X)
        # Normalize to [0, 1] range using numerically stable sigmoid
        from scipy.special import expit
        llr_norm = expit(llr)
        return llr_norm  # P(Y=1|x) proxy

    def predict(self, X, threshold=0.5):
        scores = self.score(X)
        return (scores >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────
# M3: Log-Mel ResNet-18
# ─────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Standard ResNet residual block with optional downsampling."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class LogMelResNet18(nn.Module):
    """
    M3 Baseline: ResNet-18 adapted for single-channel Log-Mel spectrogram input.
    Architecture matches Frank et al. [2021] applied to ASVspoof.
    Input:  (Batch, 1, 128, 126)
    Output: (Batch, 2) logits
    """
    def __init__(self, num_classes=2, in_channels=1, dropout=0.3):
        super().__init__()
        self.name = "M3_LogMel_ResNet18"

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.gap      = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(p=dropout)
        self.fc       = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total


# ─────────────────────────────────────────────────────────────
# M4: Lightweight RawNet2-style End-to-End Model
# ─────────────────────────────────────────────────────────────
class SincConv(nn.Module):
    """
    Sinc-based front-end convolution layer (inspired by SincNet / RawNet2).
    Learns band-pass filters directly from raw waveform.
    """
    def __init__(self, out_channels=128, kernel_size=1024, sample_rate=16000,
                 min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate  = sample_rate
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Learnable parameters: low cutoff and bandwidth
        low_hz  = 30.0 + torch.arange(out_channels).float() * (sample_rate / 2 - 30.0) / out_channels
        band_hz = torch.ones(out_channels).float() * 50.0
        self.low_hz_  = nn.Parameter(low_hz)
        self.band_hz_ = nn.Parameter(band_hz)

        # Hamming window buffer
        n = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1).float() / sample_rate
        self.register_buffer("n", n)
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(self.kernel_size).float() / (self.kernel_size - 1))
        self.register_buffer("window", window)

    def forward(self, x):
        low  = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, None]

        sinc_low  = 2 * low[:, None]  * torch.sinc(2 * low[:, None]  * self.n[None, :])
        sinc_high = 2 * high[:, None] * torch.sinc(2 * high[:, None] * self.n[None, :])

        filters = (sinc_high - sinc_low) * self.window[None, :]
        filters = filters / (2 * band)
        filters = filters.unsqueeze(1)

        return F.conv1d(x, filters, padding=self.kernel_size // 2)


class ResBlock1D(nn.Module):
    """1D residual block for raw waveform processing."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.pool  = nn.MaxPool1d(3)

    def forward(self, x):
        res = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        out = self.bn2(self.conv2(out))
        # Truncate residual if shapes differ after pooling
        out = out + res[:, :, :out.shape[-1]]
        out = F.leaky_relu(out, 0.3)
        return self.pool(out)


class LightweightRawNet2(nn.Module):
    """
    M4 Baseline: Lightweight end-to-end anti-spoofing model inspired by RawNet2.
    Tak et al. [ICASSP 2021] — processes raw waveform directly.
    Input:  (Batch, 64000) raw waveform
    Output: (Batch, 2) logits
    """
    def __init__(self, num_classes=2, sinc_channels=128, gru_hidden=1024):
        super().__init__()
        self.name = "M4_LightweightRawNet2"

        self.sinc_conv = SincConv(out_channels=sinc_channels, kernel_size=1024)
        self.bn_sinc   = nn.BatchNorm1d(sinc_channels)

        self.res_blocks = nn.ModuleList([
            ResBlock1D(sinc_channels),
            ResBlock1D(sinc_channels),
            ResBlock1D(sinc_channels),
        ])

        self.gru = nn.GRU(sinc_channels, gru_hidden, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(gru_hidden, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, T) -> (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.sinc_conv(x)
        out = F.leaky_relu(self.bn_sinc(out), 0.3)

        for block in self.res_blocks:
            out = block(out)

        # (B, C, T) -> (B, T, C) for GRU
        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        out = out[:, -1, :]  # last time step

        out = F.leaky_relu(self.fc1(self.dropout(out)), 0.3)
        return self.fc2(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
