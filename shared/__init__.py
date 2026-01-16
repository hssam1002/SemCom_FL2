"""
Shared modules for semantic communication system.

This module contains components that are shared between transmitter and receiver:
- Task Embedding: Handles task prompt/embedding (note: actual processing done by Florence-2 processor)
- CSI (Channel State Information): Contains SNR and channel type information

Main Components:
- TaskEmbedding: Task embedding module (kept for compatibility)
- CSI: Channel State Information container
  - Stores effective SNR (dB)
  - Calculates noise power/std based on SNR
  - Used by Channel module for noise generation
"""

from .task_embedding import TaskEmbedding
from .csi import CSI

__all__ = ['TaskEmbedding', 'CSI']
