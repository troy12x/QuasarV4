# c:\quasarv4\quasar\transformer_model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
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

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]

        Returns:
            output: Tensor, shape [batch_size, vocab_size]
        """
        # Get embeddings and add positional encoding
        # Transformer layers expect (seq_len, batch_size, embedding_dim)
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        # Our implementation uses batch_first=True, so shape is (batch_size, seq_len, embedding_dim)

        # Generate causal mask
        mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # The TransformerDecoder expects target and memory. For a decoder-only
        # setup, we pass the same input as both target and memory.
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=mask, memory_mask=mask)
        
        # We only need the output for the very last token for next-token prediction
        last_output = output[:, -1, :] # (batch_size, embedding_dim)
        
        # Final linear layer
        output_logits = self.output_layer(last_output)
        
        return output_logits
