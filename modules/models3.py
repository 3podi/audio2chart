import torch
import torch.nn as nn
from transformers import AutoProcessor, EncodecModel

class Encodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.codebook_dim = self.model.config.codebook_dim  # 128
        self.target_bandwidths = self.model.config.target_bandwidths  # [1.5, 3, 6, 12, 24]

    def forward(self, input_values, padding_mask, bandwidth: float = 3.0, return_embeddings: bool = False):
        """
        Encode input waveform into discrete audio codes and optionally their embedding vectors.

        Args:
            input_values (torch.Tensor): Input audio waveform of shape [batch_size, channels, sequence_length]
                                       or [channels, sequence_length] for single audio. Expected at 24 kHz.
            padding_mask (torch.Tensor): Mask of shape [batch_size, channels, sequence_length], 1 for valid, 0 for padded.
            bandwidth (float, optional): Target bandwidth in kbps (e.g., 3.0 for 3kbps). Must be in target_bandwidths.
            return_embeddings (bool, optional): If True, also return the embedding vectors. Defaults to False.

        Returns:
            tuple: If return_embeddings=False, returns (audio_codes, audio_scales, last_frame_pad_length).
                   If return_embeddings=True, returns (audio_codes, audio_scales, last_frame_pad_length, embeddings).
                   - audio_codes (torch.LongTensor): Shape [nb_frames, batch_size, nb_quantizers, frame_len].
                   - audio_scales (list): Scaling factors for each frame.
                   - last_frame_pad_length (int): Padding length in the last frame.
                   - embeddings (torch.Tensor, optional): Shape [batch_size, T_audio, codebook_dim].
        """
        # Validate bandwidth
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"Bandwidth {bandwidth} not in {self.target_bandwidths}")

        # Ensure input shape is [batch_size, channels, sequence_length]
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(0)  # Add batch dimension
        batch_size, channels, input_length = input_values.shape
        if channels not in [1, 2]:
            raise ValueError(f"Channels must be 1 or 2, got {channels}")

        # Encode waveform to get audio_codes
        encoder_outputs = self.model.encode(
            input_values,
            padding_mask,
            bandwidth=bandwidth,
            return_dict=True
        )
        audio_codes = encoder_outputs.audio_codes  # [nb_frames, batch_size, nb_quantizers, frame_len]
        audio_scales = encoder_outputs.audio_scales
        last_frame_pad_length = encoder_outputs.last_frame_pad_length or 0

        if not return_embeddings:
            return audio_codes, audio_scales, last_frame_pad_length

        # Get dimensions
        nb_frames, batch_size, nb_quantizers, frame_len = audio_codes.shape
        T_audio = nb_frames * frame_len

        # Mask padding in last frame
        if last_frame_pad_length > 0:
            audio_codes[-1, :, :, -last_frame_pad_length:] = 0

        # Reshape to [batch_size, nb_quantizers, T_audio]
        codes = audio_codes.permute(1, 2, 0, 3).reshape(batch_size, nb_quantizers, T_audio)

        # Batch embedding lookup using quantizer's codebook
        quantizer = self.model.quantizer
        all_embeds = torch.stack([layer.codebook.embed for layer in quantizer.layers[:nb_quantizers]])  # [nb_quantizers, codebook_size, codebook_dim]
        embd = nn.functional.embedding(
            codes.view(-1, T_audio),  # [batch_size * nb_quantizers, T_audio]
            all_embeds.view(-1, self.codebook_dim)  # [nb_quantizers * codebook_size, codebook_dim]
        )  # [batch_size * nb_quantizers, T_audio, codebook_dim]
        embeddings = embd.view(batch_size, nb_quantizers, T_audio, -1).sum(dim=1)  # [batch_size, T_audio, codebook_dim]

        return audio_codes, audio_scales, last_frame_pad_length, embeddings

# Example usage
if __name__ == "__main__":
    # Load processor for 24 kHz
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    
    # Sample audio (1s at 24 kHz)
    audio = torch.randn(1, 1, 24000)  # [batch_size, channels, sequence_length]
    padding_mask = torch.ones_like(audio).bool()

    # Initialize model
    model = Encodec()

    # Encode with embeddings
    audio_codes, audio_scales, last_frame_pad_length, embeddings = model(audio, padding_mask, bandwidth=3.0, return_embeddings=True)
    print(f"Audio codes shape: {audio_codes.shape}")  # Expected: [1, 1, 4, 75]
    print(f"Embeddings shape: {embeddings.shape}")  # Expected: [1, 75, 128]

    print(embeddings)
