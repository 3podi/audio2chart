import math
import inspect
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_layers import (
    PositionalEncoding,
    DecoderBlock,
)

from transformers import EncodecModel


logger = logging.getLogger(__name__)


###############################
#                             #
#                             #
#       Audio Encoder         #
#                             #
#                             #
###############################

class Encodec(nn.Module):
    def __init__(self, checkpoint="facebook/encodec_24khz", bandwidth=3.0,
                 sample_rate=24000, **kwargs):
        super().__init__()
        self.checkpoint = checkpoint
        self.bandwidth = float(bandwidth)
        self.output_kind = "codes"
        # SEANet's LSTM has no fused bf16 kernel; see WaveformTransformerDiscrete
        # ._encode_audio. 0.130s per call in fp32 against 0.712s under autocast.
        self.prefers_fp32 = True
        self.model = EncodecModel.from_pretrained(checkpoint)
        self.codebook_dim = self.model.config.codebook_dim  # 128
        self.target_bandwidths = self.model.config.target_bandwidths  # [1.5, 3, 6, 12, 24]
        model_sample_rate = self.model.config.sampling_rate
        if sample_rate != model_sample_rate:
            raise ValueError(
                f"Encodec checkpoint {checkpoint} requires {model_sample_rate} Hz, "
                f"but model.sample_rate is {sample_rate}"
            )
        if self.bandwidth not in self.target_bandwidths:
            raise ValueError(f"Bandwidth {self.bandwidth} not in {self.target_bandwidths}")
        bits_per_code = math.log2(self.model.config.codebook_size)
        frame_rate = model_sample_rate / math.prod(self.model.config.upsampling_ratios)
        self.num_quantizers = int(
            self.bandwidth * 1000 // (frame_rate * bits_per_code)
        )

    def forward(self, input_values, padding_mask, bandwidth=None, return_embeddings: bool = False):
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
        bandwidth = self.bandwidth if bandwidth is None else float(bandwidth)
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



class ResnetBlock2d(nn.Module):
    """Residual block for SEANet using Conv2d instead of Conv1d."""
    def __init__(self, dim, kernel_size=3, dilation=1, compress=2):
        super().__init__()
        hidden = dim // compress
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(
                dim, hidden,
                kernel_size=(1, kernel_size),
                padding=(0, dilation * (kernel_size - 1) // 2),
                dilation=(1, dilation)
            ),
            nn.ELU(),
            nn.Conv2d(hidden, dim, kernel_size=(1, 1))
        )

    def forward(self, x):
        return x + self.block(x)



class SEANetEncoder2d(nn.Module):
    """SEANet encoder for raw audio (Conv2d version)."""
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        in_channels=1,
        base_channels=32,
        dimension=256,
        n_residual_layers=3,
        ratios=[8, 5, 4, 2],   # stride per block
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        use_transformer=False,
        config_transformer=None
    ):
        super().__init__()

        self.ratios = ratios
        self.kernel_size = kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.n_residual_layers = n_residual_layers
        self.dilation_base = dilation_base
        self.last_kernel_size = last_kernel_size
        self.output_kind = "features"
        self.output_dim = dimension

        layers = []
        channels = base_channels

        # First conv
        layers.append(
            nn.Conv2d(
                in_channels, channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2)
            )
        )

        # Downsampling blocks
        for ratio in ratios:
            # Residual stack
            for i in range(n_residual_layers):
                layers.append(ResnetBlock2d(channels, residual_kernel_size, dilation=dilation_base**i))

            # Downsample
            layers.append(nn.ELU())
            layers.append(
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=(1, 2 * ratio),
                    stride=(1, ratio),
                    padding=(0, ratio)
                )
            )
            channels *= 2

        # Final projection
        layers.append(nn.ELU())
        layers.append(
            nn.Conv2d(
                channels, dimension,
                kernel_size=(1, last_kernel_size),
                padding=(0, last_kernel_size // 2)
            )
        )

        # Final Transformer encoder
        if use_transformer:
            layers.append(
                TransformerEncoder(
                    vocab_size=vocab_size,
                    pad_token_id=pad_token_id,
                    **config_transformer
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: waveform (B, 1, T)
        Returns:
            latent sequence (B, D, T_out)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.model(x.unsqueeze(2))
        # TransformerEncoder already returns [B, T, D].
        if x.dim() == 4:
            x = x.squeeze(2).transpose(1, 2)
        return x

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
    


###############################
#                             #
#                             #
#       Transformers          #
#                             #
#                             #
###############################


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1, use_flash=False, is_causal=False):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Token embedding and positional encoding
        if vocab_size:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
 
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, use_flash, is_causal) #is_causal=False -> EncoderBlock
            for _ in range(n_layers)
        ])
        
        # Output projection
        if vocab_size:
            self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Special init, gpt2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('linear_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(self, input_ids):
        """Create attention mask where 1 means attend, 0 means don't attend (pad token)"""
        return (input_ids != self.pad_token_id).long()
    
    def forward(self, input_ids, attention_mask=None, class_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices or Token embeddings
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            x: (batch_size, seq_len, vocab_size) - Output logits or Output embedding
        """


        # Create attention mask if not provided
        #if attention_mask is None and self.vocab_size is None:
        #    causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool)
        #    attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings and positional encoding
        if self.vocab_size:
            if attention_mask is None:
                attention_mask = self.create_attention_mask(input_ids)
            x = self.token_embedding(input_ids)
        else:
            x = input_ids.squeeze(2).permute(0,2,1)
            #if attention_mask is None:
            #    B, T, _ = x.size()
            #    attention_mask = torch.ones(B, T, device=x.device, dtype=torch.bool) 
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Output projection to vocabulary
        if self.vocab_size:
            x = self.output_projection(x)
        
        return x



class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1, conditional=False, use_flash=False, weigth_tying=False):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.conditional = conditional
        if self.conditional:
            self.cond_embedding = nn.Embedding(4, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        if weigth_tying:
            self.token_embedding.weight = self.output_projection.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        # Special init, gpt2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('linear_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(self, input_ids):
        """Create attention mask where 1 means attend, 0 means don't attend (pad token)"""
        return (input_ids != self.pad_token_id).long()
    
    def forward(self, input_ids, attention_mask=None, class_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Output logits
        """


        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings and positional encoding
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        if self.conditional:
            if class_ids is None:
                raise ValueError("class_ids are required for conditional generation")
            # Embed the class index and add to the input
            x = x + self.cond_embedding(class_ids).unsqueeze(1)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)
        
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            "Optimizer parameters: %d decayed tensors (%d parameters), "
            "%d non-decayed tensors (%d parameters)",
            len(decay_params), num_decay_params, len(nodecay_params), num_nodecay_params,
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logger.info("Using fused AdamW: %s", use_fused)

        return optimizer
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, attention_mask=None):
        """Simple greedy generation with temperature and top-k sampling"""
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                # Create or update attention mask
                if attention_mask is not None:
                    current_mask = torch.cat([
                        attention_mask, 
                        torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=torch.long)
                    ], dim=1)
                else:
                    current_mask = None
                
                # Forward pass
                logits = self.forward(generated, current_mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = current_mask
                
                # Stop if we generate pad token
                if (next_token == self.pad_token_id or next_token == self.eos_token_id).all():
                    logger.debug("Encountered the end-of-sequence token")
                    break
                    
        return generated
    



###############################
#                             #
#       Transformers          #
#     Audio Conditioned       #
#                             #
#                             #
###############################


class MERT(nn.Module):
    """Music-understanding features instead of a compression codec.

    Encodec is trained to reconstruct a waveform at low bitrate. At the 3 kbps this
    project uses it is throwing away most of what it heard, and what survives is
    whatever is cheapest to code -- not whatever identifies the line a charter follows.
    Measured, the model barely uses its audio at all: swapping in a different song's
    audio costs only +0.189 nats, and that number has not moved across the pad-weight
    fix, guidance, input noise, or stem mixing.

    MERT is trained on music with musical targets (a masked-prediction objective over
    acoustic and pitch teachers), so its features are meant to carry what the music is
    doing rather than how to rebuild the waveform.

    It is close to a drop-in here. MERT-v1-95M runs at 24 kHz, the rate the pipeline
    already uses, and its conv stack strides by 320, so it emits 75 frames a second --
    exactly Encodec's frame rate. The cross-attention sequence length is unchanged and
    no audio needs re-caching.

    LOADED WITHOUT trust_remote_code. The published repo ships custom modeling code and
    the usual recipe is AutoModel(..., trust_remote_code=True), which executes arbitrary
    Python from a downloaded repository -- the same risk Robert ruled on for
    weights_only=False ("only for a checkpoint you produced yourself, never for one that
    was downloaded"). Reading that code rather than running it shows MERTModel subclasses
    HubertModel and its only addition, MERTFeatureProjection, is identical to
    HubertFeatureProjection whenever feature_extractor_cqt is false, which it is for this
    checkpoint. So the weights are loaded into the transformers library's own audited
    HuBERT: all 211 tensors map with nothing missing and nothing unexpected, verified in
    tests/test_mert_encoder.py. If a checkpoint ever turns CQT on, this raises rather
    than silently loading a different architecture.
    """

    def __init__(self, checkpoint="m-a-p/MERT-v1-95M", sample_rate=24000,
                 layer=None, **kwargs):
        super().__init__()
        import json

        from huggingface_hub import hf_hub_download
        from transformers import HubertConfig, HubertModel

        self.output_kind = "features"
        # No recurrence here, so the fp32 forcing Encodec needs is pure cost.
        self.prefers_fp32 = False
        self.checkpoint = checkpoint

        raw = json.loads(
            Path(hf_hub_download(checkpoint, "config.json")).read_text(encoding="utf-8"))
        if raw.get("feature_extractor_cqt"):
            raise ValueError(
                f"{checkpoint} enables feature_extractor_cqt, which is the one part of "
                "MERT that is not plain HuBERT. Loading it as HuBERT would silently drop "
                "the CQT branch.")
        if int(raw.get("sample_rate", sample_rate)) != int(sample_rate):
            raise ValueError(
                f"{checkpoint} expects {raw.get('sample_rate')} Hz, pipeline is "
                f"{sample_rate} Hz")

        raw.pop("auto_map", None)
        raw.pop("architectures", None)
        config = HubertConfig(**{k: v for k, v in raw.items() if k != "model_type"})
        self.model = HubertModel.from_pretrained(checkpoint, config=config)
        self.output_dim = int(config.hidden_size)
        self.sample_rate = int(sample_rate)

        # Which layer to read. MERT's own guidance, and the MARBLE results, are that no
        # single layer is best for everything -- lower layers carry onsets and timbre,
        # middle layers carry more musical structure, and the last is specialised to the
        # pretraining objective. Rather than guess, the default learns a softmax weighting
        # over all 13 hidden states, which is 13 parameters and lets the decoder take what
        # it needs. An int pins one layer instead, for ablations.
        self.layer = layer
        if layer is None:
            self.layer_weights = nn.Parameter(
                torch.zeros(int(config.num_hidden_layers) + 1))
        else:
            self.layer_weights = None

    def trainable_when_frozen(self):
        """Parameters that must keep learning even with freeze_encoder set.

        The 95M backbone is pretrained and stays frozen. The layer blend is not: it is
        13 scalars this class adds, and freezing them makes the "learned" weighting a
        fixed uniform average.
        """
        return [] if self.layer_weights is None else [self.layer_weights]

    def forward(self, input_values):
        # The pipeline carries audio as [B, 1, T]; HuBERT wants [B, T].
        if input_values.dim() == 3:
            input_values = input_values.squeeze(1)

        # preprocessor_config.json sets do_normalize with feat_extract_norm "group", so
        # the waveform is zero-mean unit-variance per clip. Done here rather than through
        # Wav2Vec2FeatureExtractor to keep this on-GPU and out of the dataloader.
        values = input_values.float()
        mean = values.mean(dim=-1, keepdim=True)
        variance = values.var(dim=-1, keepdim=True, unbiased=False)
        values = (values - mean) / torch.sqrt(variance + 1e-7)

        output = self.model(values, output_hidden_states=True)
        if self.layer_weights is None:
            return output.hidden_states[self.layer]

        # The blend has to be built with gradients even though the backbone is frozen.
        #
        # _encode_audio wraps a frozen encoder in torch.no_grad(), which is right for the
        # 95M backbone and wrong for these 13 scalars: under no_grad they get no gradient
        # and the "learned" weighting is a fixed uniform average. Caught by running a step
        # on CPU and reading layer_weights.grad, which was None.
        #
        # Re-enabling here keeps the backbone frozen -- the stacked states arrive with
        # requires_grad False and stay detached -- while letting the multiply record a
        # graph node for the weights alone.
        with torch.enable_grad():
            stacked = torch.stack(output.hidden_states, dim=0)
            weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            return (stacked * weights).sum(dim=0)
