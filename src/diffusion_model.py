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
        
        # 特征嵌入（改为连续空间）
        self.feature_layer = nn.Linear(1, hidden_dim)
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim))
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入序列张量，形状为 [batch_size, seq_length]
            t: 时间步张量，形状为 [batch_size]
            mask: 可选的注意力掩码，形状为 [batch_size, seq_length]
        Returns:
            预测噪声，形状为 [batch_size, seq_length]
        """
        # 确保x是浮点类型并增加通道维度
        x = x.float().unsqueeze(-1)  # [batch_size, seq_length, 1]
        
        # 嵌入特征
        x = self.feature_layer(x)  # [batch_size, seq_length, hidden_dim]
        
        # 添加位置嵌入
        x = x + self.pos_embedding
        
        # 添加时间嵌入
        t_emb = self.time_mlp(t)  # [batch_size, hidden_dim]
        x = x + t_emb.unsqueeze(1)  # 时间嵌入广播到序列维度
        
        # Transformer编码
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # 输出层
        x = self.output_layer(x)  # [batch_size, seq_length, 1]
        
        return x.squeeze(-1)  # [batch_size, seq_length]

class DiffusionProcess:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从q(x_t | x_0)采样
        """
        device = x_start.device
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 确保t在正确的设备上
        if t.device != self.betas.device:
            t = t.to(self.betas.device)
            
        alphas_cumprod_t = self.alphas_cumprod[t]
        
        # 处理维度以便广播
        if x_start.dim() > 1:
            alphas_cumprod_t = alphas_cumprod_t.view(-1, *([1] * (x_start.dim() - 1)))
        
        # 确保设备一致
        alphas_cumprod_t = alphas_cumprod_t.to(device)
        
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
        从p(x_{t-1} | x_t)采样
        """
        device = x_t.device
        
        with torch.no_grad():
            # 确保t在正确的设备上
            if t.device != self.betas.device:
                t = t.to(self.betas.device)
                
            # 预测噪声
            pred_noise = model(x_t, t, mask)
            
            # 计算均值和方差
            alphas_t = self.alphas[t]
            alphas_cumprod_t = self.alphas_cumprod[t]
            
            # 处理维度以便广播
            if x_t.dim() > 1:
                alphas_t = alphas_t.view(-1, *([1] * (x_t.dim() - 1)))
                alphas_cumprod_t = alphas_cumprod_t.view(-1, *([1] * (x_t.dim() - 1)))
                betas_t = self.betas[t].view(-1, *([1] * (x_t.dim() - 1)))
            else:
                betas_t = self.betas[t]
            
            # 确保设备一致
            alphas_t = alphas_t.to(device)
            alphas_cumprod_t = alphas_cumprod_t.to(device)
            betas_t = betas_t.to(device)
            
            mean = (1. / torch.sqrt(alphas_t)) * \
                   (x_t - (betas_t / torch.sqrt(1. - alphas_cumprod_t)) * pred_noise)
            
            # 从N(mean, beta)采样
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(betas_t) * noise
            
            return x_t_minus_1
    
    def p_sample_loop(
        self,
        model: ProteinDiffusionModel,
        shape: Tuple[int, int],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        通过运行逆向过程生成样本
        """
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, mask)
            
        return x 