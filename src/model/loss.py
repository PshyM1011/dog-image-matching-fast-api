"""
Metric learning loss functions for dog image matching.
Based on ModelDevPart document - Triplet Loss and ArcFace.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    Ensures same dog embeddings are close, different dogs are far apart.
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
        if distance_metric == 'euclidean':
            self.distance_fn = self._euclidean_distance
        elif distance_metric == 'cosine':
            self.distance_fn = self._cosine_distance
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance."""
        return torch.norm(x1 - x2, p=2, dim=1)
    
    def _cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance (1 - cosine similarity)."""
        return 1 - F.cosine_similarity(x1, x2, dim=1)
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive (same dog) embeddings [batch_size, embedding_dim]
            negative: Negative (different dog) embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_dist = self.distance_fn(anchor, positive)
        neg_dist = self.distance_fn(anchor, negative)
        
        # Triplet loss: max(0, margin + pos_dist - neg_dist)
        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        
        return loss.mean()


class HardTripletLoss(nn.Module):
    """
    Hard triplet loss - uses hardest positive and hardest negative in batch.
    More effective for training.
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def _pairwise_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        if self.distance_metric == 'euclidean':
            return torch.cdist(x1, x2, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            x1_norm = F.normalize(x1, p=2, dim=1)
            x2_norm = F.normalize(x2, p=2, dim=1)
            similarity = torch.matmul(x1_norm, x2_norm.t())
            return 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard triplet loss.
        
        Args:
            embeddings: All embeddings [batch_size, embedding_dim]
            labels: Labels for each embedding [batch_size]
            
        Returns:
            Hard triplet loss value
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings for stable training
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        distance_matrix = self._pairwise_distance(embeddings, embeddings)
        
        # Create mask for positive pairs (same label)
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Find hardest positive (largest distance among positives)
        positive_distances = distance_matrix * positive_mask
        # Set non-positive pairs to a very negative value (not -1, which could be confused with valid 0)
        positive_distances[positive_mask == 0] = -999.0  # Ignore non-positive pairs
        hardest_positive, _ = positive_distances.max(dim=1)
        
        # Find hardest negative (smallest distance among negatives)
        negative_distances = distance_matrix * negative_mask
        negative_distances[negative_mask == 0] = float('inf')  # Ignore non-negative pairs
        hardest_negative, _ = negative_distances.min(dim=1)
        
        # Only compute loss where we have valid triplets
        # Valid if: has positive pair (hardest_positive > -999) AND has negative pair (hardest_negative < inf)
        valid_mask = (hardest_positive > -999.0) & (hardest_negative < float('inf'))
        
        if valid_mask.sum() == 0:
            # No valid triplets - this shouldn't happen if batch has multiple classes
            # Return a loss that encourages learning
            # Use average distance as a proxy loss
            avg_distance = distance_matrix.mean()
            return torch.clamp(avg_distance * 0.1, min=0.01, max=1.0)
        
        # Compute triplet loss only for valid triplets
        # Loss = max(0, margin + positive_distance - negative_distance)
        raw_loss = self.margin + hardest_positive[valid_mask] - hardest_negative[valid_mask]
        
        # Clamp to [0, inf] - but don't let it be exactly 0
        loss_values = torch.clamp(raw_loss, min=1e-6)
        
        # If loss is too small (all easy triplets), use a scaled version
        # This ensures the model still learns to separate embeddings
        mean_loss = loss_values.mean()
        
        if mean_loss < 0.05:
            # All triplets are too easy - use a scaled loss based on actual distances
            # This encourages the model to push negatives further away
            distance_diff = hardest_negative[valid_mask] - hardest_positive[valid_mask]
            # Use a soft loss that's proportional to how close negatives are to positives
            soft_loss = torch.clamp(
                (self.margin - distance_diff) * 0.5,
                min=0.05,  # Minimum meaningful loss
                max=2.0    # Cap to prevent explosion
            )
            return soft_loss.mean()
        
        return mean_loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for metric learning.
    Adds angular margin to improve feature discrimination.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0
    ):
        """
        Initialize ArcFace loss.
        
        Args:
            embedding_dim: Embedding dimension
            num_classes: Number of dog classes
            margin: Angular margin (in radians)
            scale: Feature scale
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            
        Returns:
            ArcFace loss value
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)  # [batch_size, num_classes]
        
        # Convert to angles
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add margin to target class
        target_theta = theta.gather(1, labels.unsqueeze(1))
        target_theta_margin = target_theta + self.margin
        
        # Compute new cosine with margin
        target_cosine = torch.cos(target_theta_margin)
        
        # Replace target class cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output = (one_hot * target_cosine) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Cross entropy loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Triplet + ArcFace for better training.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        triplet_margin: float = 1.0,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        triplet_weight: float = 0.5,
        arcface_weight: float = 0.5
    ):
        super().__init__()
        self.triplet_loss = HardTripletLoss(margin=triplet_margin)
        self.arcface_loss = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale  # Reduced from default 64.0 to prevent very high loss
        )
        self.triplet_weight = triplet_weight
        self.arcface_weight = arcface_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Returns:
            Total loss and loss components dictionary
        """
        triplet = self.triplet_loss(embeddings, labels)
        arcface = self.arcface_loss(embeddings, labels)
        
        total_loss = self.triplet_weight * triplet + self.arcface_weight * arcface
        
        loss_dict = {
            'total': total_loss,
            'triplet': triplet,
            'arcface': arcface
        }
        
        return total_loss, loss_dict

