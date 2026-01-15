"""
Channel models for semantic communication.
Implements Noiseless, AWGN, and Rayleigh fading channels.
"""

import torch
import torch.nn as nn
from typing import Optional

from shared.csi import CSI


class Channel(nn.Module):
    """
    Channel model for semantic communication.
    
    Supports three channel types:
    - Noiseless: Perfect channel without noise
    - AWGN: Additive White Gaussian Noise
    - Rayleigh: Rayleigh fading with AWGN
    
    Args:
        csi: Channel State Information (contains SNR and channel type)
    """
    
    def __init__(self, csi: CSI):
        super().__init__()
        self.csi = csi
        self.channel_type = csi.channel_type.lower()
        
        if self.channel_type not in ['noiseless', 'awgn', 'rayleigh']:
            raise ValueError(
                f"Unsupported channel type: {self.channel_type}. "
                "Supported types: 'noiseless', 'awgn', 'rayleigh'"
            )
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Transmit signal through channel.
        
        Args:
            signal: Input signal tensor
            
        Returns:
            Received signal after channel transmission
        """
        if self.channel_type == 'noiseless':
            return self._noiseless(signal)
        elif self.channel_type == 'awgn':
            return self._awgn(signal)
        elif self.channel_type == 'rayleigh':
            return self._rayleigh(signal)
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")
    
    def _noiseless(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Noiseless channel (perfect transmission).
        
        Args:
            signal: Input signal
            
        Returns:
            Signal unchanged
        """
        return signal
    
    def _awgn(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Additive White Gaussian Noise channel.
        
        Args:
            signal: Input signal
            
        Returns:
            Signal with AWGN added
        """
        # Calculate signal power (assume normalized)
        signal_power = torch.mean(signal ** 2)
        
        # Get noise standard deviation from CSI
        noise_std = self.csi.get_noise_std(signal_power.item())
        
        # Generate AWGN
        noise = torch.randn_like(signal) * noise_std
        
        # Add noise to signal
        received_signal = signal + noise
        
        return received_signal
    
    def _rayleigh(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Rayleigh fading channel with AWGN.
        
        Args:
            signal: Input signal
            
        Returns:
            Signal after Rayleigh fading and AWGN
        """
        # Calculate signal power
        signal_power = torch.mean(signal ** 2)
        
        # Generate Rayleigh fading coefficients
        # Rayleigh fading: h = sqrt(0.5) * (h_real + j*h_imag)
        # where h_real and h_imag are independent Gaussian RVs
        # For real-valued signals, we use |h| = sqrt(h_real^2 + h_imag^2)
        h_real = torch.randn_like(signal)
        h_imag = torch.randn_like(signal)
        fading_coeff = torch.sqrt(h_real ** 2 + h_imag ** 2) * (0.5 ** 0.5)
        
        # Apply fading
        faded_signal = fading_coeff * signal
        
        # Get noise standard deviation
        noise_std = self.csi.get_noise_std(signal_power.item())
        
        # Generate AWGN
        noise = torch.randn_like(signal) * noise_std
        
        # Add noise
        received_signal = faded_signal + noise
        
        return received_signal
    
    def update_csi(self, new_csi: CSI):
        """
        Update Channel State Information.
        
        Args:
            new_csi: New CSI object
        """
        self.csi = new_csi
        self.channel_type = new_csi.channel_type.lower()


def create_channel(
    channel_type: str = 'awgn',
    effective_snr_db: float = 20.0
) -> Channel:
    """
    Factory function to create a channel.
    
    Args:
        channel_type: Type of channel ('noiseless', 'awgn', 'rayleigh')
        effective_snr_db: Effective SNR in dB
        
    Returns:
        Channel instance
    """
    csi = CSI(effective_snr_db=effective_snr_db, channel_type=channel_type)
    return Channel(csi)
