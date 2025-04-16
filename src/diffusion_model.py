import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ProteinDiffusionModel(nn.Module):
    def __init__(
        self,
        seq_length: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Amino acid embedding
        self.aa_embedding = nn.Embedding(21, hidden_dim)  # 20 amino acids + padding
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim))
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 21)  # Predict amino acid probabilities
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence tensor of shape [batch_size, seq_length]
            t: Time step tensor of shape [batch_size]
            mask: Optional attention mask of shape [batch_size, seq_length]
        Returns:
            Predicted noise of shape [batch_size, seq_length, 21]
        """
        # Embed amino acids
        x = self.aa_embedding(x)  # [batch_size, seq_length, hidden_dim]
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Add time embeddings
        t_emb = self.time_mlp(t)  # [batch_size, hidden_dim]
        x = x + t_emb.unsqueeze(1)  # Broadcast time embeddings across sequence
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Output layer
        x = self.output_layer(x)  # [batch_size, seq_length, 21]
        
        return x

class DiffusionProcess:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t = alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1)
        
        x_t = torch.sqrt(alphas_cumprod_t) * x_start + \
              torch.sqrt(1. - alphas_cumprod_t) * noise
              
        return x_t, noise
    
    def p_sample(
        self,
        model: ProteinDiffusionModel,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        """
        with torch.no_grad():
            # Predict noise
            pred_noise = model(x_t, t, mask)
            
            # Calculate mean and variance
            alphas_t = self.alphas[t]
            alphas_cumprod_t = self.alphas_cumprod[t]
            
            alphas_t = alphas_t.unsqueeze(-1).unsqueeze(-1)
            alphas_cumprod_t = alphas_cumprod_t.unsqueeze(-1).unsqueeze(-1)
            
            mean = (1. / torch.sqrt(alphas_t)) * \
                   (x_t - (self.betas[t].unsqueeze(-1).unsqueeze(-1) / \
                    torch.sqrt(1. - alphas_cumprod_t)) * pred_noise)
            
            # Sample from N(mean, beta)
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(self.betas[t].unsqueeze(-1).unsqueeze(-1)) * noise
            
            return x_t_minus_1
    
    def p_sample_loop(
        self,
        model: ProteinDiffusionModel,
        shape: Tuple[int, int],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples by running the reverse process
        """
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, mask)
            
        return x 