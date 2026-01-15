"""
Channel module for semantic communication.
Implements various channel models: Noiseless, AWGN, and Rayleigh.
"""

from .channel import Channel, create_channel

__all__ = ['Channel', 'create_channel']
