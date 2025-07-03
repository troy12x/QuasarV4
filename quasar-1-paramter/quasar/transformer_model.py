# c:\quasarv4\quasar\transformer_model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input tensor. Supports batch_first."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape: (1, max_len, d_model) for batch_first=True
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    A simple decoder-only Transformer model for causal language modeling.
    """
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=nlayers)
        
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

        # Tie the weights of the embedding layer and the output layer
        self.output_layer.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor, shape [batch_size, seq_len]

        Returns:
            output: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        # 1. Get embeddings
        src_emb = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        
        # 2. Add positional encoding
        src_pos = self.pos_encoder(src_emb)
        
        # 3. Create causal mask
        mask = self._generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)

        # 4. Pass through Transformer decoder
        # The decoder is used in a self-attention manner, so tgt and memory are the same.
        output = self.transformer_decoder(tgt=src_pos, memory=src_pos, tgt_mask=mask, memory_mask=mask)
        
        # 5. Final output layer
        output = self.output_layer(output)
        return output
