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
    
    Two modes supported:
    - Mode 1: Compression after Vision Tower (default)
      Processes up to vision_tower output (1024-dim features)
    - Mode 2: Compression after Image Projection Norm
      Processes up to image_proj_norm output (768-dim features)
    
    Note: All Florence-2 modules (vision_tower, etc.) are FROZEN.
          Only future compression module will be trainable.
    
    Future structure:
    - vision_tower (frozen) → compression_module (trainable) → output
    
    Args:
        florence2_model: Florence-2 model instance (frozen)
        mode: Processing mode ('vision_tower' or 'image_proj_norm')
        task_embedding_dim: Task embedding dimension (for compatibility, not used)
    """
    
    def __init__(
        self,
        florence2_model: Florence2Model,
        mode: str = 'vision_tower',
        task_embedding_dim: int = 768
    ):
        super().__init__()
        
        self.florence2_model = florence2_model
        self.mode = mode
        self.task_embedding_dim = task_embedding_dim
        
        # Get vision encoder output dimension
        self.vision_dim = florence2_model.get_vision_dim()  # 1024 for base model
        
        # Set output dimension based on mode
        if mode == 'vision_tower':
            # Mode 1: Output vision_tower features (1024-dim)
            self.output_dim = self.vision_dim  # 1024
        elif mode == 'image_proj_norm':
            # Mode 2: Output after image_proj_norm (768-dim)
            self.projected_dim = 768
            self.output_dim = self.projected_dim  # 768
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'vision_tower' or 'image_proj_norm'")
        
        print(f"Transmitter initialized with mode: {mode} (output_dim: {self.output_dim})")
    
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
        Process images through vision encoder.
        
        Mode 1 (vision_tower): Stops at vision_tower output (1024-dim)
        Mode 2 (image_proj_norm): Processes up to image_proj_norm output (768-dim)
        
        Args:
            images: Input images - can be:
                - PIL Image or list of PIL Images
                - torch.Tensor of shape (batch_size, 3, H, W) - already preprocessed
                If PIL Image, will use processor to preprocess
            
        Returns:
            Mode 1: Vision tower features (batch_size, seq_len, 1024)
            Mode 2: Processed vision features (batch_size, seq_len, 768)
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
        
        # Step 1: vision_tower.forward_features_unpool (FROZEN - use no_grad for memory efficiency)
        # Note: vision_tower is frozen (requires_grad=False), but no_grad() saves memory
        #       by preventing computation graph creation for frozen modules
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            with torch.no_grad():
                vision_output = model.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f'invalid image shape {pixel_values.shape}')
        
        # Extract features from vision_output (could be tuple or tensor)
        # Note: This is done outside no_grad() context to allow gradient flow if trainable module follows
        if isinstance(vision_output, tuple):
            vision_features = vision_output[0]  # (batch, seq_len, 1024)
        else:
            vision_features = vision_output
        
        # Future: Compression module (trainable) will be inserted here:
        #   vision_features = self.compression_module(vision_features)  # Trainable, no no_grad()
        
        # Mode 1: Stop at vision_tower output
        if self.mode == 'vision_tower':
            return vision_features  # (batch, seq_len, 1024)
        
        # Mode 2: Continue processing up to image_proj_norm
        elif self.mode == 'image_proj_norm':
            # All following steps are FROZEN - use no_grad() for memory efficiency
            # Note: These are frozen modules (requires_grad=False), but no_grad() saves memory
            with torch.no_grad():
                # Step 2: image_pos_embed (if applicable)
                T = 1  # Single frame
                if hasattr(model, 'image_pos_embed') and model.image_pos_embed is not None:
                    x = vision_features.view(batch_size * T, -1, vision_features.shape[-1])
                    num_tokens = x.shape[-2]
                    h = w = int(num_tokens ** 0.5)
                    
                    if h * w == num_tokens:
                        x = x.view(batch_size * T, h, w, x.shape[-1])
                        pos_embed = model.image_pos_embed(x)
                        x = x + pos_embed
                        x = x.view(batch_size, T * h * w, x.shape[-1])
                        vision_features = x
                
                # Step 3: visual_temporal_embed (if applicable)
                if hasattr(model, 'visual_temporal_embed') and model.visual_temporal_embed is not None:
                    x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
                    first_token = x_reshaped[:, :, 0]
                    visual_temporal_emb = model.visual_temporal_embed(first_token)
                    x_reshaped = x_reshaped + visual_temporal_emb.view(1, T, 1, vision_features.shape[-1])
                    vision_features = x_reshaped.view(batch_size, T * x_reshaped.shape[2], vision_features.shape[-1])
                
                # Step 4: image_feature_source_pooling
                x_reshaped = vision_features.view(batch_size, T, -1, vision_features.shape[-1])
                
                if hasattr(model, 'image_feature_source'):
                    image_feature_source = model.image_feature_source
                else:
                    image_feature_source = ['last_frame']
                
                x_feat_dict = {
                    'spatial_avg_pool': x_reshaped.mean(dim=2),
                    'temporal_avg_pool': x_reshaped.mean(dim=1),
                    'last_frame': x_reshaped[:, -1]
                }
                
                new_x = []
                for _image_feature_source in image_feature_source:
                    if _image_feature_source not in x_feat_dict:
                        raise ValueError(f'invalid image feature source: {_image_feature_source}')
                    new_x.append(x_feat_dict[_image_feature_source])
                
                vision_features = torch.cat(new_x, dim=1)
                
                # Step 5: image_projection (1024 -> 768)
                vision_features = vision_features @ model.image_projection
                
                # Step 6: image_proj_norm
                vision_features = model.image_proj_norm(vision_features)
            
            # Future: Compression module (trainable) will be inserted here if needed:
            #   vision_features = self.compression_module(vision_features)  # Trainable, no no_grad()
            
            return vision_features  # (batch, seq_len, 768)
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def get_output_dim(self) -> int:
        """
        Get output dimension of transmitter.
        
        Returns:
            Output dimension:
            - Mode 1 (vision_tower): 1024
            - Mode 2 (image_proj_norm): 768
        """
        return self.output_dim
    
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
        
        # Calculate number of patches
        # DaViT typically uses patch size 16
        patch_size = 16
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        return (batch_size, num_patches, output_dim)
        
