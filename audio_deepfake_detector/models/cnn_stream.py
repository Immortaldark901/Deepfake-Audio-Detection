"""
Stream B: Lightweight 4-Block 2D Convolutional Neural Network (CNN).
Input: Log-Mel Spectrogram of shape (Batch, 1, 128, 126).
Supports feature extraction, classification, and Grad-CAM hooks on the final convolutional layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard Conv2D -> BatchNorm2d -> ReLU -> MaxPool2d block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class Lightweight4BlockCNN(nn.Module):
    """
    Compact 4-Block CNN for spectrogram artifact and spectral evidence detection.
    
    Channel progression: 1 -> 16 -> 32 -> 64 -> 128
    Spatial reduction: (128, 126) -> (64, 63) -> (32, 31) -> (16, 15) -> (8, 7)
    Global Average Pooling -> 128-dimensional embedding.
    """
    def __init__(self, in_channels=1, num_classes=2, embed_dim=128, return_embeddings=False):
        super().__init__()
        self.return_embeddings = return_embeddings
        self.embed_dim = embed_dim
        
        self.block1 = ConvBlock(in_channels, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.block4 = ConvBlock(64, embed_dim)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Internal storage for Grad-CAM activations & gradients
        self.last_conv_act = None
        self.last_conv_grad = None

    def _hook_grad(self, grad):
        self.last_conv_grad = grad

    def forward(self, x, return_embed=None, retain_cam_maps=False):
        """
        Args:
            x (torch.Tensor): Spectrogram tensor of shape (Batch, 1, 128, 126)
            return_embed (bool, optional): Override embedding return flag.
            retain_cam_maps (bool): Register backward hook for Grad-CAM extraction.
            
        Returns:
            torch.Tensor: Logits (Batch, 2) or Embedding (Batch, 128)
        """
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)  # Shape: (Batch, 128, 8, 7)
        
        if retain_cam_maps:
            self.last_conv_act = x4
            if x4.requires_grad:
                x4.register_hook(self._hook_grad)
                
        # Global Average Pooling
        pooled = self.gap(x4).flatten(1)  # (Batch, 128)
        embed = self.dropout(pooled)
        
        should_return_embed = return_embed if return_embed is not None else self.return_embeddings
        if should_return_embed:
            return embed
            
        logits = self.classifier(embed)
        return logits
