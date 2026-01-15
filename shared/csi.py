"""
Channel State Information (CSI) module.
Contains Effective SNR information shared between transmitter and receiver.
"""

import torch
from typing import Optional


class CSI:
    """
    Channel State Information container.
    Stores Effective SNR and other channel parameters.
    
    Args:
        effective_snr_db: Effective SNR in dB
        channel_type: Type of channel ('noiseless', 'awgn', 'rayleigh')
    """
    
    def __init__(
        self,
        effective_snr_db: float = 20.0,
        channel_type: str = 'awgn'
    ):
        self.effective_snr_db = effective_snr_db
        self.channel_type = channel_type
        self.snr_linear = 10 ** (effective_snr_db / 10.0)
    
    def get_noise_power(self, signal_power: Optional[float] = None) -> float:
        """
        Calculate noise power based on SNR.
        
        Args:
            signal_power: Signal power (if None, assumes normalized signal)
            
        Returns:
            Noise power (variance)
        """
        if signal_power is None:
            # Assume normalized signal power = 1
            signal_power = 1.0
        
        noise_power = signal_power / self.snr_linear
        return noise_power
    
    def get_noise_std(self, signal_power: Optional[float] = None) -> float:
        """
        Calculate noise standard deviation.
        
        Args:
            signal_power: Signal power (if None, assumes normalized signal)
            
        Returns:
            Noise standard deviation
        """
        noise_power = self.get_noise_power(signal_power)
        return (noise_power ** 0.5)
    
    def update_snr(self, new_snr_db: float):
        """
        Update SNR value.
        
        Args:
            new_snr_db: New SNR value in dB
        """
        self.effective_snr_db = new_snr_db
        self.snr_linear = 10 ** (new_snr_db / 10.0)
    
    def __repr__(self) -> str:
        return f"CSI(effective_snr_db={self.effective_snr_db:.2f}, channel_type='{self.channel_type}')"
