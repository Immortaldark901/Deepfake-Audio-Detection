"""
Stream A: LFCC Multi-Layer Perceptron (MLP) Classifier / Embedding Backbone.
Input: 120-dimensional LFCC feature vector (Static, Delta, Delta-Delta means and stds).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFCC_MLP(nn.Module):
    """
    Lightweight Multi-Layer Perceptron for LFCC feature discrimination.
    Architecture:
      - Linear(120, 64) -> BatchNorm1d -> LeakyReLU(0.2) -> Dropout(0.3)
      - Linear(64, 32) -> BatchNorm1d -> LeakyReLU(0.2) -> Dropout(0.2)
      - Linear(32, num_classes) [if classifier head enabled]
    """
    def __init__(self, input_dim=120, hidden_dim=64, embed_dim=32, num_classes=2, return_embeddings=False):
        super().__init__()
        self.return_embeddings = return_embeddings
        
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.2)
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embed=None):
        """
        Args:
            x (torch.Tensor): LFCC vector of shape (Batch, 120)
            return_embed (bool, optional): Override embedding return flag.
            
        Returns:
            torch.Tensor: Logits (Batch, 2) or Embedding (Batch, 32)
        """
        h1 = self.block1(x)
        embed = self.block2(h1)
        
        should_return_embed = return_embed if return_embed is not None else self.return_embeddings
        if should_return_embed:
            return embed
            
        logits = self.classifier(embed)
        return logits
