"""
Dual-view encoder combining CNN (EfficientNet) and ViT (Vision Transformer).
Based on ModelDevPart document - CNN for local textures, ViT for global structure.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Optional


class DualViewEncoder(nn.Module):
    """
    Dual-view encoder that fuses CNN and ViT features.
    
    Architecture:
    - EfficientNet-B0: Extracts local texture features (nose, fur, eye shape)
    - ViT-B/16: Extracts global structure features (body shape, silhouette)
    - Fusion: Concatenates both embeddings and projects to final dimension
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        cnn_dim: int = 1280,  # EfficientNet-B0 output dim
        vit_dim: int = 768,   # ViT-B/16 output dim
        dropout: float = 0.3,
        use_pretrained: bool = True
    ):
        """
        Initialize dual-view encoder.
        
        Args:
            embedding_dim: Final embedding dimension (default: 512)
            cnn_dim: CNN output dimension
            vit_dim: ViT output dimension
            dropout: Dropout rate
            use_pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # CNN branch (EfficientNet-B0) - Local texture features
        if use_pretrained:
            self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.cnn = efficientnet_b0(weights=None)
        
        # Remove classification head, keep feature extractor
        self.cnn.classifier = nn.Identity()
        
        # ViT branch (Vision Transformer-B/16) - Global structure features
        if use_pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)
        
        # Remove classification head
        self.vit.heads = nn.Identity()
        
        # Fusion layer
        fusion_dim = cnn_dim + vit_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # L2 normalization for metric learning
        self.normalize = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual-view encoder.
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            Embedding tensor [batch_size, embedding_dim]
        """
        # Extract CNN features (local textures)
        cnn_features = self.cnn(x)  # [batch_size, 1280]
        
        # Extract ViT features (global structure)
        vit_features = self.vit(x)  # [batch_size, 768]
        
        # Fuse features
        fused = torch.cat([cnn_features, vit_features], dim=1)  # [batch_size, 2048]
        
        # Project to embedding space
        embedding = self.fusion(fused)  # [batch_size, 512]
        
        # L2 normalize for cosine similarity
        if self.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only CNN features (for analysis)."""
        return self.cnn(x)
    
    def extract_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only ViT features (for analysis)."""
        return self.vit(x)


class FrontalEncoder(nn.Module):
    """
    Encoder specifically for frontal view images.
    Focuses on facial features: muzzle, nose, eyes, forehead patterns.
    """
    
    def __init__(self, embedding_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        if use_pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        self.backbone.classifier = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.projection(features)
        return nn.functional.normalize(embedding, p=2, dim=1)


class LateralEncoder(nn.Module):
    """
    Encoder specifically for lateral view images.
    Focuses on body features: shape, coat color, torso patterns, tail.
    """
    
    def __init__(self, embedding_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        if use_pretrained:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.backbone = vit_b_16(weights=None)
        
        self.backbone.heads = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.projection(features)
        return nn.functional.normalize(embedding, p=2, dim=1)


class DualViewFusionModel(nn.Module):
    """
    Complete dual-view model that processes frontal and lateral views separately
    and fuses them into a single embedding.
    """
    
    def __init__(
        self,
        frontal_encoder: Optional[nn.Module] = None,
        lateral_encoder: Optional[nn.Module] = None,
        embedding_dim: int = 512
    ):
        """
        Initialize dual-view fusion model.
        
        Args:
            frontal_encoder: Encoder for frontal views (default: FrontalEncoder)
            lateral_encoder: Encoder for lateral views (default: LateralEncoder)
            embedding_dim: Final fused embedding dimension
        """
        super().__init__()
        
        self.frontal_encoder = frontal_encoder or FrontalEncoder(embedding_dim=256)
        self.lateral_encoder = lateral_encoder or LateralEncoder(embedding_dim=256)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(
        self,
        frontal: torch.Tensor,
        lateral: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with separate frontal and lateral inputs.
        
        Args:
            frontal: Frontal view images [batch_size, 3, 224, 224]
            lateral: Lateral view images [batch_size, 3, 224, 224]
            
        Returns:
            Fused embedding [batch_size, embedding_dim]
        """
        frontal_emb = self.frontal_encoder(frontal)
        lateral_emb = self.lateral_encoder(lateral)
        
        # Concatenate embeddings
        combined = torch.cat([frontal_emb, lateral_emb], dim=1)
        
        # Fuse
        fused = self.fusion(combined)
        
        # Normalize
        return nn.functional.normalize(fused, p=2, dim=1)
    
    def encode_frontal(self, frontal: torch.Tensor) -> torch.Tensor:
        """Encode only frontal view."""
        return self.frontal_encoder(frontal)
    
    def encode_lateral(self, lateral: torch.Tensor) -> torch.Tensor:
        """Encode only lateral view."""
        return self.lateral_encoder(lateral)

