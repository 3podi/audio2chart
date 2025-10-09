import torch
import torch.nn as nn
from transformers import AutoProcessor, EncodecModel


class Encodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_48khz")
        self.codebook_dim = self.model.config.codebook_dim        

    def forward(self, input_values, padding_mask, bandwidth: float = 3.0):
        """
        Process input waveform to produce audio embeddings using EnCodec's quantizer.

        Args:
            waveform (torch.Tensor): Input audio waveform of shape [batch_size, sequence_length]
                                    or [sequence_length] for single audio.
            bandwidth (float, optional): Target bandwidth in kbps (e.g., 6.0 for 6kbps).
                                        Must be in model.config.target_bandwidths.

        Returns:
            torch.Tensor: Audio embeddings of shape [batch_size, T_audio, embd_dim],
                         where T_audio = nb_frames * frame_len, embd_dim = codebook_dim (128).
        """
        # Encode waveform to get audio_codes
        encoder_outputs = self.model.encode(
            input_values,
            padding_mask,
            bandwidth=bandwidth,
            return_dict=True
        )
        audio_codes = encoder_outputs.audio_codes  # [nb_frames, batch_size, nb_quantizers, frame_len]
        last_frame_pad_length = encoder_outputs.last_frame_pad_length or 0

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
        embd = embd.view(batch_size, nb_quantizers, T_audio, -1).sum(dim=1)  # [batch_size, T_audio, codebook_dim]

        return embd  # [batch_size, T_audio, codebook_dim=128]
