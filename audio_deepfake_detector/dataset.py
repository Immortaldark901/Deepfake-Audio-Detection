"""
Dataset Loader and Protocol Parser for ASVspoof 2019 Logical Access (LA).
Supports Train, Dev, and Evaluation partitions with per-attack metadata tracking (A01–A19).
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Try importing audio libraries with fallbacks
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.io import wavfile
    HAS_SCIPY_WAV = True
except ImportError:
    HAS_SCIPY_WAV = False


def load_audio_file(file_path, target_sr=16000, max_samples=64000):
    """
    Robust audio file loader with multi-library fallback.
    Converts to mono, resamples to 16 kHz, and crops/pads to max_samples.
    """
    waveform = None
    sr = target_sr
    
    if HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
        except Exception:
            waveform = None
            
    if waveform is None and HAS_SOUNDFILE:
        try:
            data, sr = sf.read(file_path, dtype='float32')
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data)
        except Exception:
            waveform = None
            
    if waveform is None and HAS_SCIPY_WAV and file_path.endswith('.wav'):
        try:
            sr, data = wavfile.read(file_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data)
        except Exception:
            waveform = None

    if waveform is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Audio file does not exist: {file_path}")
        raise RuntimeError(f"[Error] Failed to load audio file with all available backends: {file_path}")

    # Pad or crop to target sample length (64,000 samples = 4.0s @ 16kHz)
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    elif waveform.shape[0] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[0]))
        
    return waveform


class ASVspoof2019LADataset(Dataset):
    """
    ASVspoof 2019 Logical Access (LA) Dataset.
    
    Protocol format:
    SPEAKER_ID  AUDIO_ID  ENVIRONMENT_ID  ATTACK_ID  KEY
    e.g. LA_0079 LA_T_1138215 - - bonafide
         LA_0079 LA_T_1271820 - A01 spoof
    """
    def __init__(self, protocol_file, audio_dir, max_len=64000, feature_extractor=None, transform=None):
        self.protocol_file = protocol_file
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.samples = []

        if os.path.exists(protocol_file):
            self._load_protocol()
        else:
            print(f"[Warning] Protocol file not found: {protocol_file}")

    def _load_protocol(self):
        with open(self.protocol_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                speaker_id = parts[0]
                audio_id = parts[1]
                env_id = parts[2]
                attack_id = parts[3]
                key = parts[4]
                
                # Binary label mapping: bonafide -> 0, spoof -> 1
                label = 0 if key.lower() == "bonafide" else 1
                
                # Check for audio file existence
                audio_path = os.path.join(self.audio_dir, f"{audio_id}.flac")
                if not os.path.exists(audio_path):
                    audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")
                    
                self.samples.append({
                    "speaker_id": speaker_id,
                    "audio_id": audio_id,
                    "env_id": env_id,
                    "attack_id": attack_id,
                    "label": label,
                    "path": audio_path
                })
        print(f"[Dataset] Loaded {len(self.samples)} entries from {self.protocol_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav = load_audio_file(item["path"], target_sr=16000, max_samples=self.max_len)
        
        if self.transform is not None:
            wav = self.transform(wav)
            
        label = torch.tensor(item["label"], dtype=torch.long)
        
        if self.feature_extractor is not None:
            with torch.no_grad():
                lfcc, mel = self.feature_extractor(wav.unsqueeze(0))
            return {
                "waveform": wav,
                "lfcc": lfcc.squeeze(0),
                "mel": mel.squeeze(0),
                "label": label,
                "audio_id": item["audio_id"],
                "attack_id": item["attack_id"]
            }
            
        return {
            "waveform": wav,
            "label": label,
            "audio_id": item["audio_id"],
            "attack_id": item["attack_id"]
        }


class SyntheticDemoDataset(Dataset):
    """
    Synthetic / Mock Dataset for unit testing, offline verification,
    and rapid pipeline benchmarking without raw flac files.
    """
    def __init__(self, num_samples=100, max_len=64000, spoof_ratio=0.5):
        self.num_samples = num_samples
        self.max_len = max_len
        self.samples = []
        
        for i in range(num_samples):
            is_spoof = (i / num_samples) < spoof_ratio
            label = 1 if is_spoof else 0
            attack_id = f"A{np.random.randint(7, 20):02d}" if is_spoof else "-"
            self.samples.append({
                "audio_id": f"SYNTH_{i:06d}",
                "attack_id": attack_id,
                "label": label
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.samples[idx]
        t = torch.linspace(0, 4.0, self.max_len)
        
        # Bona fide: clean harmonics (e.g. 220 Hz + 440 Hz)
        if item["label"] == 0:
            wav = 0.5 * torch.sin(2 * np.pi * 220 * t) + 0.3 * torch.sin(2 * np.pi * 440 * t)
        else:
            # Spoof: high frequency artifacts and glitch noise
            wav = 0.4 * torch.sin(2 * np.pi * 220 * t) + 0.3 * torch.sin(2 * np.pi * 5500 * t) + 0.05 * torch.randn(self.max_len)
            
        wav = wav.float()
        label = torch.tensor(item["label"], dtype=torch.long)
        
        return {
            "waveform": wav,
            "label": label,
            "audio_id": item["audio_id"],
            "attack_id": item["attack_id"]
        }


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
    """
    Constructs a robust DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
