"""
Image utility functions.
"""

import torch
from PIL import Image
from typing import Union, Tuple, Optional
import numpy as np


def load_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = (224, 224)
) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        target_size: Target size (H, W) for resizing (optional)
        
    Returns:
        PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    
    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    
    return image


def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Preprocess image for Florence-2 model.
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        normalize: Whether to normalize to [0, 1] or ImageNet stats
        device: Device to place tensor on
        
    Returns:
        Preprocessed image tensor of shape (1, 3, H, W)
    """
    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        # Convert PIL to numpy
        image = np.array(image).astype(np.float32) / 255.0
        # Convert to tensor and permute to (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        if image.dim() == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        if image.max() > 1.0:
            image = image / 255.0
        if image.dim() == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Normalize if requested (ImageNet normalization)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std
    
    # Move to device
    if device is not None:
        image = image.to(device)
    
    return image
