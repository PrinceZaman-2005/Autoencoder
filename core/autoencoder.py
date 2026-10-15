import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Architecture:
        Input -> Encoder -> Latent Representation -> Decoder -> Reconstruction

    The encoder compresses the input into a smaller representation.
    The decoder uses that representation to reconstruct the input.
    """

    def __init__(self, input_size=64, latent_size=16):
        super().__init__()

        # Encoder: compress the input.
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size),
            nn.ReLU()
        )

        # Decoder: reconstruct the input.
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Pass input through the encoder and decoder.

        Args:
            x: Input tensor with shape (batch_size, input_size).

        Returns:
            Reconstructed input with the same shape as x.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return reconstruction

    def encode(self, x):
        """Return the latent representation of the input."""
        return self.encoder(x)

    def decode(self, latent):
        """Reconstruct input from a latent representation."""
        return self.decoder(latent)
