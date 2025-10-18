import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_layers import (
    PositionalEncoding,
    DecoderBlock,
    DecoderBlockCrossAttention
)


###############################
#                             #
#                             #
#       Audio Encoder         #
#                             #
#                             #
###############################

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
        kernel_size=8,
        last_kernel_size=8,
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

        #self.model = nn.Sequential(*layers)
        self.layers=layers

    def forward(self, x):
        """
        Args:
            x: waveform (B, 1, T)
        Returns:
            latent sequence (B, D, T_out)
        """
        print('starting forward in the audio encoder.')
        for layer in self.layers:
            print('running: ', layer)
            x = layer(x)
        return x
        #return self.model(x)
    
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
        # Add dummy spatial dimension: (B, C, T) -> (B, C, 1, T)
        x = x.unsqueeze(1)
        x = self.model(x.unsqueeze(1))    # (B, D, 1, T_out)
        return x#.squeeze(2)  # (B, D, T_out)

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
            assert class_ids is not None, "class_idx must be provided for conditional transformer"
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

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
                    print('Found exit token, stopping generation')
                    break
                    
        return generated
    



###############################
#                             #
#       Transformers          #
#     Audio Conditioned       #
#                             #
#                             #
###############################



class TransformerDecoderAudioConditioned(nn.Module):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=512, max_audio_len=10000, dropout=0.1, conditional=False, use_flash=False, weigth_tying=False):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.audio_positional_encoding = PositionalEncoding(d_model, max_audio_len)
        self.conditional = conditional
        if self.conditional:
            self.cond_embedding = nn.Embedding(4, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlockCrossAttention(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])

        # Audio adapter
        self.adapter = nn.Conv1d(
            in_channels=128,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False
        )
        
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
    
    def forward(self, input_ids, input_audio, attention_mask=None, class_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Output logits
        """

        # Process audio
        #input_audio = input_audio.permute(0, 2, 1)
        #input_audio = self.adapter(input_audio) 
        #print('input audio shape iniziale: ', input_audio.shape)
        input_audio = self.audio_positional_encoding(input_audio)#.permute(0, 2, 1))
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings and positional encoding
        #print('x shape: ', input_ids.shape)
        x = self.token_embedding(input_ids)
        #print('x shape after token embed: ', x.shape)
        x = self.positional_encoding(x)
        #print('x shape after pos enc: ', x.shape)

        if self.conditional:
            assert class_ids is not None, "class_idx must be provided for conditional transformer"
            # Embed the class index and add to the input
            #print(self.cond_embedding(class_ids).unsqueeze(1).shape)
            x = x + self.cond_embedding(class_ids)
        
        #print('shape input ids: ', x.shape)
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(decoder_input=x, encoder_output=input_audio, decoder_mask=attention_mask)
        
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    # TODO: generate/inference
    #def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, attention_mask=None):

        
