"""
Utility functions for semantic communication system.

This module provides utility functions for image processing and other common tasks.

Main Functions:
- load_image: Load image from file path with optional resizing
- preprocess_image: Preprocess image for Florence-2 model (normalization, tensor conversion)
"""

from .image_utils import load_image, preprocess_image

__all__ = ['load_image', 'preprocess_image']
