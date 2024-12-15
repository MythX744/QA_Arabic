import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffAttn(nn.Module):
    def __init__(self, dim_model):
        """
        Initialize the Difference Attention module.

        Args:
            dim_model: The input and output dimension of the module
        """
        super().__init__()

        # Initialize the weight matrices
        self.W_q = nn.Linear(dim_model, dim_model)
        self.W_k = nn.Linear(dim_model, dim_model)
        self.W_v = nn.Linear(dim_model, dim_model)

        # Initialize scalar as learnable parameter
        self.scalar = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, hidden_states):
        """
        Forward pass of the Difference Attention.

        Args:
            X: Input tensor of shape [batch_size, seq_length, dim_model]

        Returns:
            Attention output of shape [batch_size, seq_length, dim_model]
        """
        # Compute and split Q into Q1 and Q2
        Q = self.W_q(hidden_states)
        Q1, Q2 = torch.chunk(Q, chunks=2, dim=-1)

        # Compute and split K into K1 and K2
        K = self.W_k(hidden_states)
        K1, K2 = torch.chunk(K, chunks=2, dim=-1)

        # Compute V
        V = self.W_v(hidden_states)

        # Get dimension for scaling
        d = V.shape[-1]

        # Computing attention scaling factor
        s = 1 / math.sqrt(d)

        # Compute attention scores
        A1 = Q1 @ K1.transpose(-1, -2) * s
        A2 = Q2 @ K2.transpose(-1, -2) * s

        # Calculate the difference attention
        attention = (F.softmax(A1, dim=-1) - self.scalar * F.softmax(A2, dim=-1)) @ V

        # Return the difference attention
        return attention


class MultiHead(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.heads = nn.ModuleList([DiffAttn(dim_model) for _ in range(num_heads)])
        self.W_o = nn.Linear(dim_model * num_heads, dim_model)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False):
        """
        Match GPT2's attention interface
        Args:
            hidden_states: input tensor (same as X in original implementation)
            layer_past: unused, kept for compatibility
            attention_mask: unused, kept for compatibility
            head_mask: unused, kept for compatibility
            use_cache: unused, kept for compatibility
            output_attentions: unused, kept for compatibility
        """
        # Process through each attention head
        outputs = []
        for head in self.heads:
            outputs.append(head(hidden_states))

        # Concatenate all heads
        out = torch.cat(outputs, dim=-1)

        # Apply output projection
        out = self.W_o(out)

        # Scale by (1 - λ)
        out = out * (1 - self.heads[0].scalar)

        # Apply LayerNorm and return
        return self.norm(out), None
