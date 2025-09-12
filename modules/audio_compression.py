import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    """Residual block for SEANet."""
    def __init__(self, dim, kernel_size=3, dilation=1, compress=2):
        super().__init__()
        hidden = dim // compress
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(dim, hidden, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(hidden, dim, 1),
        )
    
    def forward(self, x):
        return x + self.block(x)

class SEANetEncoder(nn.Module):
    """SEANet encoder for raw audio."""
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        dimension=128,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],   # stride per block
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
    ):
        super().__init__()

        self.ratios = ratios 
        self.kernel_size = kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.n_residual_layers = n_residual_layers
        self.dilation_base = dilation_base
        self.last_kernel_size = last_kernel_size

        layers = []
        channels = base_channels

        # First conv
        layers.append(nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size//2))

        # Downsampling blocks
        for ratio in ratios:
            # Residual stack
            for i in range(n_residual_layers):
                layers.append(ResnetBlock(channels, residual_kernel_size, dilation=dilation_base**i))
            
            # Downsample
            layers.append(nn.ELU())
            layers.append(
                nn.Conv1d(
                    channels,
                    channels * 2,
                    kernel_size=2*ratio,
                    stride=ratio,
                    padding=ratio
                )
            )
            channels *= 2

        # Final projection
        layers.append(nn.ELU())
        layers.append(nn.Conv1d(channels, dimension, last_kernel_size, padding=last_kernel_size//2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: waveform (B, 1, T)
        Returns:
            latent sequence (B, D, T_out)
        """
        return self.model(x)
    
    def compute_receptive_field(self, sr=16000):
        """
        Compute total stride, receptive field (samples & ms), and compression ratio.
        """
        stride_total = 1
        rf = self.kernel_size  # first conv
        
        for ratio in self.ratios:
            # Residual layers at this stage
            for j in range(self.n_residual_layers):
                dilation = self.dilation_base**j
                rf += (self.residual_kernel_size - 1) * dilation * stride_total
            
            # Downsampling conv
            rf += (2*ratio - 1) * stride_total
            stride_total *= ratio

        # Final conv
        rf += (self.last_kernel_size - 1) * stride_total

        rf_ms = rf / sr * 1000
        return dict(
            stride_total=stride_total,
            receptive_field_samples=rf,
            receptive_field_ms=rf_ms
        )