import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from tqdm import tqdm
from .moe import MoELayer

class QuasarConfig(PretrainedConfig):
    model_type = "quasar"

    def __init__(
        self,
        vocab_size=129280,
        embedding_dim=8192,
        num_hidden_layers=96,
        num_attention_heads=64,
        num_experts=128,
        expert_dim=2048,
        top_k=4,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.top_k = top_k
        super().__init__(**kwargs)

class SelfAttention(nn.Module):
    def __init__(self, config: QuasarConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // self.num_heads
        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.k_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.v_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)

class QuasarBlock(nn.Module):
    def __init__(self, config: QuasarConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.moe_layer = MoELayer(
            embedding_dim=config.embedding_dim,
            num_experts=config.num_experts,
            expert_dim=config.expert_dim,
            top_k=config.top_k
        )
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        moe_out, lb_loss = self.moe_layer(self.ln2(x))
        x = x + moe_out
        return x, lb_loss

class Quasar(PreTrainedModel):
    config_class = QuasarConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: QuasarConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        print(f"\nInitializing {config.num_hidden_layers} Quasar layers...")
        self.layers = nn.ModuleList([QuasarBlock(config) for _ in tqdm(range(config.num_hidden_layers), desc="Creating Quasar Layers")])
        self.final_ln = nn.LayerNorm(config.embedding_dim)
        self.output_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        total_lb_loss = 0.0

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, lb_loss = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), x, use_reentrant=False)
            else:
                x, lb_loss = layer(x)
            total_lb_loss += lb_loss

        x = self.final_ln(x)
        logits = self.output_head(x)

        loss = None
        if labels is not None:
            main_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = main_loss + total_lb_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'lb_loss': total_lb_loss
        }
