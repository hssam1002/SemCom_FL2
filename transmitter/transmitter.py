"""
Transmitter module.
Processes images through Florence-2 vision encoder up to vision_tower.
Compression point: AFTER_VISION_TOWER (Method 2)

Based on Florence-2's _encode_image method, but stops at vision_tower output.
The remaining processing (pos_embed, temporal_embed, pooling, projection, norm)
is done at the Receiver side.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from models.florence2_model import Florence2Model


class Transmitter(nn.Module):
    """
    Transmitter for semantic communication.
    
    Method 2: Compression after Vision Tower (Recommended)
    Processes images through Florence-2 vision encoder up to vision_tower output.
    Compression will be applied to vision_tower output (1024-dim features).
    
    Flow:
    1. Image processing (preprocessing)
    2. vision_tower.forward_features_unpool(pixel_values)
    → Compression point (for future compression module)
    
    Remaining steps (done at Receiver):
    - image_pos_embed
    - visual_temporal_embed
    - image_feature_source pooling
    - image_projection
    - image_proj_norm
    - language_model
    
    Args:
        florence2_model: Florence-2 model instance
        task_embedding_dim: Task embedding dimension (for compatibility, not used)
        include_linear_embedding: Whether to include linear embedding (for compatibility, not used)
        use_pooled_features: Whether to use pooled features (deprecated, kept for compatibility)
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
        self.vision_dim = florence2_model.get_vision_dim()  # 1024 for base model
        
        # Output dimension: vision_tower output (1024 for base model)
        # This is the compression point (Method 2)
        self.output_dim = self.vision_dim  # 1024
        
        if include_linear_embedding:
            print("Warning: include_linear_embedding is ignored. Using vision_tower output (1024 dim).")
        self.linear_embedding = None
    
    def encode_task_prompts(
        self,
        task_prompts: Union[List[str], torch.Tensor],
        images: Optional[Union[torch.Tensor, List, "PIL.Image.Image"]] = None
    ) -> torch.Tensor:
        """
        Encode task prompts to text embeddings.
        
        NOTE: This method is kept for backward compatibility.
        It is recommended to generate text_embeddings at top level (main/test_semcom.py)
        using processor with all-zero dummy image, and share them with Tx/Rx.
        
        Args:
            task_prompts: Task prompt strings
            images: Images (required for processor to tokenize correctly)
                    If None, will use dummy image
        
        Returns:
            Text embeddings: (batch, text_seq_len, 768)
        """
        processor = self.florence2_model.processor
        model = self.florence2_model.model
        
        if isinstance(task_prompts, list):
            # Use processor with images to get correct tokenization
            # processor(text, images) converts task prompts correctly
            if images is None:
                # Create dummy image if not provided
                from PIL import Image as PILImage
                import numpy as np
                dummy_image = PILImage.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
                if not isinstance(task_prompts, list):
                    images = [dummy_image]
                else:
                    images = [dummy_image] * len(task_prompts)
            
            inputs = processor(
                text=task_prompts,
                images=images if isinstance(images, list) else [images],
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(
                device=next(model.parameters()).device,
                dtype=torch.long
            )
            embedding_layer = model.get_input_embeddings()
            text_embeddings = embedding_layer(input_ids)
            return text_embeddings
        else:
            # Already embeddings
            return task_prompts
    
    def forward(
        self,
        images: Union[torch.Tensor, List, "PIL.Image.Image"]
    ) -> torch.Tensor:
        """
        Process images through vision encoder up to vision_tower output.
        
        Method 2: Compression after Vision Tower
        1. Use processor to preprocess images (resize, normalize, etc.)
        2. vision_tower.forward_features_unpool
        → Output: Vision tower features ready for compression
        
        Args:
            images: Input images - can be:
                - PIL Image or list of PIL Images
                - torch.Tensor of shape (batch_size, 3, H, W) - already preprocessed
                If PIL Image, will use processor to preprocess
            
        Returns:
            Vision tower features (1024 dimension for base model)
            Shape: (batch_size, seq_len, 1024)
            Note: seq_len depends on image size (typically 576 for 768x768 image)
        """
        model = self.florence2_model.model
        processor = self.florence2_model.processor
        
        # Preprocess images using processor if needed
        if not isinstance(images, torch.Tensor):
            # images is PIL Image or list of PIL Images
            # Use processor to preprocess (this ensures proper resize/crop for square feature maps)
            if not isinstance(images, list):
                images = [images]
            
            # Process images using processor (same as Florence-2 does)
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            
            # Move to device and dtype
            device = next(model.parameters()).device
            pixel_values = pixel_values.to(device=device, dtype=model.dtype)
        else:
            # Already a tensor, just ensure dtype matches
            pixel_values = images
            if pixel_values.dtype != model.dtype:
                pixel_values = pixel_values.to(dtype=model.dtype)
        
        # Step 1: vision_tower.forward_features_unpool
        # Method 2: Stop here (compression point)
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            vision_output = model.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        
        # Extract features from vision_output (could be tuple or tensor)
        if isinstance(vision_output, tuple):
            vision_features = vision_output[0]  # (batch, seq_len, 1024)
        else:
            vision_features = vision_output
        
        # Output: Vision tower features ready for compression
        # Shape: (batch_size, seq_len, 1024)
        # Note: Remaining processing (pos_embed, temporal_embed, pooling, projection, norm)
        #       will be done at the Receiver side
        return vision_features
    
    def get_output_dim(self) -> int:
        """
        Get output dimension of transmitter.
        
        Returns:
            Output dimension (1024 for vision_tower output, base model)
        """
        return self.output_dim  # 1024 for vision_tower output (base model)
    
    def get_output_shape(self, batch_size: int = 1, image_size: Tuple[int, int] = (224, 224)) -> Tuple[int, ...]:
        """
        Get output shape of transmitter.
        
        Args:
            batch_size: Batch size
            image_size: Image size (H, W)
            
        Returns:
            Output shape tuple (batch_size, num_patches, 1024)
        """
        output_dim = self.get_output_dim()  # 1024
        
        if self.use_pooled_features:
            return (batch_size, output_dim)
        else:
            # Calculate number of patches
            # DaViT typically uses patch size 16
            patch_size = 16
            num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
            return (batch_size, num_patches, output_dim)
