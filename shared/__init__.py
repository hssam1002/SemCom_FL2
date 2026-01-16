"""
Shared modules for semantic communication system.

This module contains components that are shared between transmitter and receiver:
- CSI (Channel State Information): Contains SNR and channel type information

Main Components:
- CSI: Channel State Information container
  - Stores effective SNR (dB)
  - Calculates noise power/std based on SNR
  - Used by Channel module for noise generation

Note: Task embedding is handled by Florence-2's processor directly.
      TaskEmbedding class exists for compatibility but is not actively used.
"""

from .csi import CSI

__all__ = ['CSI']
