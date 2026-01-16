"""
Transmitter module for semantic communication.

This module implements the transmitter (Tx) component of the semantic communication system.
The transmitter processes images through Florence-2's vision encoder up to the vision_tower output.

Architecture: Method 2 (Compression after Vision Tower)
- Processes images through vision_tower
- Output: Vision tower features (1024-dim for base model)
- Compression point: After vision_tower output

Main Components:
- Transmitter: Main transmitter class that processes images to vision embeddings
"""

from .transmitter import Transmitter

__all__ = ['Transmitter']
