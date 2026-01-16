"""
Models module for Florence-2 and related components.

This module provides a wrapper for the Florence-2 model from HuggingFace.
It handles model loading, device management, and provides convenient access to model components.

Main Components:
- Florence2Model: Wrapper class for Florence-2 model
  - Handles model loading from HuggingFace
  - Provides access to vision_tower, language_model, processor
  - Manages device placement and dtype
- get_vision_encoder_output_dim: Utility function to get vision encoder output dimension
  - base: 1024 (DaViT-base)
  - large: 1024 (DaViT-large)
"""

from .florence2_model import Florence2Model, get_vision_encoder_output_dim

__all__ = ['Florence2Model', 'get_vision_encoder_output_dim']
