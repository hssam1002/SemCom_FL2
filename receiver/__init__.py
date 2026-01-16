"""
Receiver module for semantic communication.

This module implements the receiver (Rx) component of the semantic communication system.
The receiver processes received vision_tower features through the remaining Florence-2 pipeline.

Architecture: Method 2 (Compression after Vision Tower)
- Receives vision_tower features from transmitter
- Processes through: pos_embed, temporal_embed, pooling, projection, norm
- Merges with text embeddings (shared from top level)
- Generates text using language_model

Main Components:
- Receiver: Main receiver class that processes vision features and generates text
"""

from .receiver import Receiver

__all__ = ['Receiver']
