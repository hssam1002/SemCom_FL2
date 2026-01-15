"""
Transmitter module.
Processes images through Florence-2 vision encoder and optionally applies linear embedding.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.florence2_model import Florence2Model


class Transmitter(nn.Module):
    """
    Transmitter for semantic communication.
    
    Processes images through Florence-2 vision encoder and optionally
    applies linear embedding to match task embedding dimension.
    
    Args:
        florence2_model: Florence-2 model instance
        task_embedding_dim: Dimension of task embedding
        include_linear_embedding: Whether to include linear embedding layer
                                 to match task embedding dimension
        use_pooled_features: Whether to use pooled features (CLS token) 
                             instead of full sequence
    """
    
    def __init__(
        self,
        florence2_model: Florence2Model,
        task_embedding_dim: int = 768,
        include_linear_embedding: bool = False,
        use_pooled_features: bool = False
    ):
        super().__init__()
        
        self.florence2_model = florence2_model
        self.task_embedding_dim = task_embedding_dim
        self.include_linear_embedding = include_linear_embedding
        self.use_pooled_features = use_pooled_features
        
        # Get vision encoder output dimension
        self.vision_dim = florence2_model.get_vision_dim()
        
        # Optional linear embedding to match task embedding dimension
        if include_linear_embedding:
            if use_pooled_features:
                # If using pooled features, input is (batch, vision_dim)
                self.linear_embedding = nn.Linear(self.vision_dim, task_embedding_dim)
            else:
                # If using full sequence, input is (batch, seq_len, vision_dim)
                # We'll apply linear to the last dimension
                self.linear_embedding = nn.Linear(self.vision_dim, task_embedding_dim)
        else:
            self.linear_embedding = None
    
    def forward(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Process images through vision encoder (DaViT).
        
        This is the Transmitter part: Image -> Vision Encoder -> Vision Embedding
        
        Args:
            images: Input images tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Encoded vision features (vision embedding)
            - If use_pooled_features=True: (batch_size, task_embedding_dim) or (batch_size, vision_dim)
            - If use_pooled_features=False: (batch_size, seq_len, task_embedding_dim) or (batch_size, seq_len, vision_dim)
        """
        # Encode images using Florence-2 vision encoder
        vision_features, pooled_features = self.florence2_model.encode_image(
            images,
            return_pooled=True
        )
        
        # Choose between full sequence or pooled features
        if self.use_pooled_features:
            features = pooled_features  # (batch_size, vision_dim)
        else:
            features = vision_features  # (batch_size, seq_len, vision_dim)
        
        # Apply linear embedding if enabled
        if self.include_linear_embedding and self.linear_embedding is not None:
            if self.use_pooled_features:
                # (batch_size, vision_dim) -> (batch_size, task_embedding_dim)
                features = self.linear_embedding(features)
            else:
                # (batch_size, seq_len, vision_dim) -> (batch_size, seq_len, task_embedding_dim)
                features = self.linear_embedding(features)
        
        return features
    
    def get_output_dim(self) -> int:
        """
        Get output dimension of transmitter.
        
        Returns:
            Output dimension
        """
        if self.include_linear_embedding:
            return self.task_embedding_dim
        else:
            return self.vision_dim
    
    def get_output_shape(self, batch_size: int = 1, image_size: Tuple[int, int] = (224, 224)) -> Tuple[int, ...]:
        """
        Get output shape of transmitter.
        
        Args:
            batch_size: Batch size
            image_size: Image size (H, W)
            
        Returns:
            Output shape tuple
        """
        output_dim = self.get_output_dim()
        
        if self.use_pooled_features:
            return (batch_size, output_dim)
        else:
            # Calculate number of patches
            # DaViT typically uses patch size 16
            patch_size = 16
            num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
            return (batch_size, num_patches, output_dim)
