import math
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match number of query heads in GQA."""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    """Apply rotary positional embeddings to Q and K."""
    q_embed = (q * cos) + (torch.roll(q, shifts=1, dims=-1) * sin)
    k_embed = (k * cos) + (torch.roll(k, shifts=1, dims=-1) * sin)
    return q_embed, k_embed


class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            num_kv_heads=None,
            dropout=0.1,
            bias=True,
            depth=0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        # Initialize projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize differential parameters
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)

        # Layer normalization for subspace
        self.subln = RMSNorm(2 * self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_pos=None, attention_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        # Compute Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention computation
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # Apply rotary embeddings if provided
        if rel_pos is not None:
            cos, sin = rel_pos
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepare for attention
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)

        # Scale query
        q *= self.scaling

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        # Apply causal mask if no mask provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((tgt_len, src_len), float('-inf'), device=x.device),
                diagonal=1
            )

        # Apply attention mask
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nan_to_num(attn_weights)

        # Compute differential attention weights
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        # Compute lambdas
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Apply differential attention
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # Apply attention to values
        attn = torch.matmul(attn_weights, v)

        # Apply subspace normalization and scaling
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        # Reshape and project output
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn = self.out_proj(attn)

        return attn


class MultiHead(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"

        self.dim_model = dim_model
        self.num_heads = num_heads

        # Use improved differential attention
        self.diff_attn = MultiheadDiffAttn(
            embed_dim=dim_model,
            num_heads=num_heads,
            num_kv_heads=num_heads // 2,  # Using GQA with 2:1 ratio
            dropout=0.1
        )

        self.norm = nn.LayerNorm(dim_model)

    def forward(self, hidden_states, attention_mask=None, layer_past=None,
                head_mask=None, use_cache=False, output_attentions=False):
        batch_size, seq_length = hidden_states.size()[:2]

        if attention_mask is not None:
            # Correct attention_mask shape
            attention_mask = attention_mask.squeeze(1)  # Remove the extra singleton dimension
            attention_mask = attention_mask[:, 0, :]  # Select the first sequence-related mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Expand dimensions

        # Process with differential attention
        attn_output = self.diff_attn(hidden_states, attention_mask=attention_mask)

        # Add residual connection and normalize
        output = self.norm(attn_output + hidden_states)

        outputs = (output,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)

        return outputs


def replace_gpt2_attn(model, dim_model, num_heads):
    """Replace GPT-2 attention layers with improved differential attention."""
    for i, block in enumerate(model.transformer.h):
        block.attn = MultiHead(dim_model, num_heads)
    return model
