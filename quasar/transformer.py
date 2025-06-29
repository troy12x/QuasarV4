import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 100
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_heads, 
            dim_feedforward=config.dim_feedforward, 
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.num_layers)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)

        src = self.embedding(input_ids) * math.sqrt(self.config.hidden_size)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, mask=causal_mask)
        
        logits = self.proj_out(output)
        return logits
