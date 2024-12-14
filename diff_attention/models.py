import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model
from typing import Optional, Tuple


class DifferentialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q1, Q2, K1, K2, V
        self.q_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size)  # Doubled for Q1 and Q2
        self.k_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size)  # Doubled for K1 and K2
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Learnable parameters for λ
        self.lambda_param = nn.Parameter(torch.tensor(0.8))  # Initialize λ to 0.8

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Project queries and keys twice, and values once
        q_combined = self.q_proj(hidden_states)  # [batch, seq, 2*hidden]
        k_combined = self.k_proj(hidden_states)  # [batch, seq, 2*hidden]
        v = self.v_proj(hidden_states)  # [batch, seq, hidden]

        # Split into Q1, Q2 and K1, K2
        q1, q2 = torch.chunk(q_combined, 2, dim=-1)
        k1, k2 = torch.chunk(k_combined, 2, dim=-1)

        # Reshape for multi-head attention
        def reshape_for_attention(x):
            return x.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        q1 = reshape_for_attention(q1)
        q2 = reshape_for_attention(q2)
        k1 = reshape_for_attention(k1)
        k2 = reshape_for_attention(k2)
        v = reshape_for_attention(v)

        # Compute attention scores
        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_scores1 = attn_scores1 + attention_mask
            attn_scores2 = attn_scores2 + attention_mask

        # Compute differential attention
        attn_weights1 = torch.softmax(attn_scores1, dim=-1)
        attn_weights2 = torch.softmax(attn_scores2, dim=-1)
        attn_weights = attn_weights1 - self.lambda_param * attn_weights2

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPT2QAModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()

    def replace_attention_layers(self, differential_ratio: float = 0.3):
        """
        Replace a portion of the attention layers with differential attention.

        Args:
            differential_ratio (float): Ratio of attention layers to replace (0 to 1)
        """
        # Get all attention layers
        attention_layers = [block.attn for block in self.transformer.h]
        num_layers = len(attention_layers)

        # Calculate number of layers to replace
        num_replace = int(num_layers * differential_ratio)

        # Replace attention layers starting from the top
        for i in range(num_layers - num_replace, num_layers):
            # Create new differential attention layer with same config
            diff_attn = DifferentialAttention(self.transformer.h[i].attn.config)

            # Get original weights from GPT-2 attention
            original_weights = self.transformer.h[i].attn.c_attn.weight.data
            original_bias = self.transformer.h[i].attn.c_attn.bias.data

            # In GPT-2, the weights are concatenated [q, k, v] with shape [3*n_embd, n_embd]
            n_embd = self.config.n_embd
            q_weights = original_weights[:n_embd, :]
            k_weights = original_weights[n_embd:2 * n_embd, :]
            v_weights = original_weights[2 * n_embd:3 * n_embd, :]

            # Initialize the split Q and K projections
            with torch.no_grad():
                # Initialize Q projections (both Q1 and Q2)
                diff_attn.q_proj.weight.data = torch.cat([q_weights, q_weights], dim=0)
                diff_attn.q_proj.bias.data = torch.cat([
                    original_bias[:n_embd],
                    original_bias[:n_embd]
                ])

                # Initialize K projections (both K1 and K2)
                diff_attn.k_proj.weight.data = torch.cat([k_weights, k_weights], dim=0)
                diff_attn.k_proj.bias.data = torch.cat([
                    original_bias[n_embd:2 * n_embd],
                    original_bias[n_embd:2 * n_embd]
                ])

                # Initialize V projection
                diff_attn.v_proj.weight.data = v_weights
                diff_attn.v_proj.bias.data = original_bias[2 * n_embd:3 * n_embd]

                # Initialize output projection
                diff_attn.out_proj.weight.data = self.transformer.h[i].attn.c_proj.weight.data
                diff_attn.out_proj.bias.data = self.transformer.h[i].attn.c_proj.bias.data

            # Replace the attention layer
            self.transformer.h[i].attn = diff_attn

        return self

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ) -> Tuple:
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)


def create_qa_model(pretrained_model_name: str, differential_ratio: float = 0.3):
    """
    Create a QA model with differential attention.

    Args:
        pretrained_model_name: Name of pretrained model
        differential_ratio: Ratio of attention layers to replace

    Returns:
        GPT2QAModel: Model ready for training
    """
    model = GPT2QAModel.from_pretrained(pretrained_model_name)
    model.replace_attention_layers(differential_ratio)
    return model
