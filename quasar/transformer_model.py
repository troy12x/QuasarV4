# c:\quasarv4\quasar\transformer_model.py

import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

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
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, nlayers, dropout=0.5, use_return_dict=True):
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

        self.use_return_dict = use_return_dict

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            input_ids: Tensor, shape [batch_size, seq_len]
            labels: Optional tensor of shape [batch_size, seq_len] for loss calculation.
            return_dict: Whether to return a `ModelOutput` object.

        Returns:
            CausalLMOutputWithPast or tuple
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # 1. Get embeddings
        src_emb = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        
        # 2. Add positional encoding
        src_pos = self.pos_encoder(src_emb)
        
        # 3. Create causal mask
        mask = self._generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)

        # 4. Pass through Transformer decoder
        transformer_output = self.transformer_decoder(tgt=src_pos, memory=src_pos, tgt_mask=mask, memory_mask=mask)
        
        # 5. Final output layer
        logits = self.output_layer(transformer_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            vocab_size = self.output_layer.out_features
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            if loss is not None:
                return (loss, logits)
            return logits

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None, # Not implemented
            hidden_states=None, # Not implemented
            attentions=None, # Not implemented
        )
