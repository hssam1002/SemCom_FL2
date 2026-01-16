"""
Channel module for semantic communication.

This module implements various channel models for simulating wireless communication:
- Noiseless: Perfect channel without noise (for baseline testing)
- AWGN: Additive White Gaussian Noise channel
- Rayleigh: Rayleigh fading channel with AWGN

Main Components:
- Channel: Base channel class that applies noise/fading to signals
- create_channel: Factory function to create channel instances
- CSI: Channel State Information (SNR, channel type) - imported from shared module
"""

from .channel import Channel, create_channel

__all__ = ['Channel', 'create_channel']
