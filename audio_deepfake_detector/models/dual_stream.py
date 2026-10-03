"""
Dual-Stream Acoustic Deepfake Detector Network.
Fuses Stream A (LFCC-MLP) and Stream B (4-Block CNN) to exploit complementary
linear-frequency cepstral and psychoacoustic time-frequency evidence.
Total parameters: < 0.5M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfcc_mlp import LFCC_MLP
from .cnn_stream import Lightweight4BlockCNN


class DualStreamDetector(nn.Module):
    """
    Lightweight Dual-Stream Deepfake Detector.
    
    Architecture:
      Stream A: LFCC (120) -> LFCC_MLP -> 32-dim embedding
      Stream B: Log-Mel (1, 128, 126) -> 4-Block CNN -> 128-dim embedding
      Fusion: Concat(32 + 128 = 160) -> Linear(160, 64) -> BN -> ReLU -> Dropout(0.2) -> Linear(64, 2)
    """
    def __init__(self, lfcc_dim=120, mel_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        
        # Stream A (LFCC-MLP Backbone) — classifier head is bypassed; only embedding used
        self.stream_a = LFCC_MLP(
            input_dim=lfcc_dim,
            hidden_dim=64,
            embed_dim=32,
            return_embeddings=True
        )
        # Freeze the dormant sub-classifier head (never used in fused mode)
        for p in self.stream_a.classifier.parameters():
            p.requires_grad = False
        
        # Stream B (Lightweight 4-Block CNN Backbone) — classifier head is bypassed
        self.stream_b = Lightweight4BlockCNN(
            in_channels=mel_channels,
            embed_dim=128,
            return_embeddings=True
        )
        # Freeze the dormant sub-classifier head
        for p in self.stream_b.classifier.parameters():
            p.requires_grad = False
        
        # Multimodal Fusion & Classification Head (sole classifier in fused mode)
        fusion_in_dim = 32 + 128  # 160
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, lfcc, mel, retain_cam_maps=False):
        """
        Forward pass of the dual-stream network.
        
        Args:
            lfcc (torch.Tensor): LFCC feature tensor of shape (Batch, 120)
            mel (torch.Tensor): Log-Mel spectrogram tensor of shape (Batch, 1, 128, 126)
            retain_cam_maps (bool): Retain CNN feature maps & gradient hooks for Grad-CAM.
            
        Returns:
            torch.Tensor: Classification logits of shape (Batch, 2)
        """
        # Extract stream embeddings
        embed_a = self.stream_a(lfcc)
        embed_b = self.stream_b(mel, retain_cam_maps=retain_cam_maps)
        
        # Feature Concatenation
        fused = torch.cat([embed_a, embed_b], dim=1)
        
        # Classification Logits
        logits = self.fusion_head(fused)
        return logits

    def predict_proba(self, lfcc, mel):
        """
        Computes posterior probability P(Y=1 | x) for spoof detection.
        
        Returns:
            torch.Tensor: Spoof probabilities of shape (Batch,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(lfcc, mel)
            probs = F.softmax(logits, dim=-1)
            # Class 1 is spoof (as defined in paper: Y=1 denotes spoofed speech)
            return probs[:, 1]

    def count_parameters(self):
        """
        Returns total and trainable parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self):
        """
        Prints structured architecture and parameter breakdown.
        """
        total_p, train_p = self.count_parameters()
        stream_a_p = sum(p.numel() for p in self.stream_a.parameters())
        stream_b_p = sum(p.numel() for p in self.stream_b.parameters())
        fusion_p = sum(p.numel() for p in self.fusion_head.parameters())
        
        report = (
            "=" * 55 + "\n"
            "   DUAL-STREAM AUDIO DEEPFAKE DETECTOR SUMMARY\n"
            "=" * 55 + "\n"
            f" Stream A (LFCC-MLP) Parameters    : {stream_a_p:>10,}\n"
            f" Stream B (4-Block CNN) Parameters : {stream_b_p:>10,}\n"
            f" Fusion Head Parameters            : {fusion_p:>10,}\n"
            "-" * 55 + "\n"
            f" TOTAL PARAMETERS                  : {total_p:>10,} ({total_p/1e6:.3f}M)\n"
            f" TRAINABLE PARAMETERS              : {train_p:>10,} ({train_p/1e6:.3f}M)\n"
            f" Budget Compliance (< 0.5M)        : {'PASS [VERIFIED]' if total_p < 500000 else 'FAIL'}\n"
            "=" * 55
        )
        return report
