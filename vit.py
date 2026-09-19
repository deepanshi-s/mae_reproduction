import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, 
            num_emb, 
            head_size, 
            causal_attn_flag=False, 
            use_rope=False,
        ):
        super().__init__()
        self.key = nn.Linear(num_emb, head_size)
        self.query = nn.Linear(num_emb, head_size)
        self.value = nn.Linear(num_emb, head_size)
        self.head_size = head_size

    def forward(self, x: tensor, use_cache=True):
        query = self.query(x)
        keys = self.key(x)
        value = self.value(x)
        
        B, T, C = query.shape
        b1, t1, c1 = keys.shape

        tril = torch.ones(T, t1, device=x.device)
        
        attn = F.scaled_dot_product_attention(query,keys,value, tril.bool())
        return attn


class multiAttention(nn.Module):
    def __init__(self, 
            num_heads, 
            head_size, 
            num_emb, 
            flash_attn=False, 
        ):
        super().__init__()
        self.attention = nn.ModuleList(
            [Attention(num_emb=num_emb, 
                    head_size=head_size, 
                ) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, num_emb)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(num_emb)

    def forward(self, x):
        x1 = self.layernorm(x)
        x1 = torch.cat([block(x1) for block in self.attention], -1)
        x1 = x + self.linear(x1)
        return x1


class FeedForward(nn.Module):
    def __init__(self, num_emb):
        super().__init__()
        self.layernorm = nn.LayerNorm(num_emb)
        self.block = nn.Sequential(
            nn.Linear(num_emb, num_emb * 4),
            nn.ReLU(),
            nn.Linear(num_emb * 4, num_emb),
        )

    def forward(self, x):
        out = self.layernorm(x)
        out = self.block(out)
        return x + out


class Block(nn.Module):
    def __init__(self, 
            num_heads, 
            head_size, 
            num_emb, 
        ):
        super().__init__()        
        self.mha = multiAttention(num_heads, 
                        head_size, 
                        num_emb, 
                    )
        self.linear = FeedForward(num_emb)

    def forward(self, x):
        x1 = self.mha(x)
        out = self.linear(x1)
        return out