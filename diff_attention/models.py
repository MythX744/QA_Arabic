import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model
from typing import Optional, Tuple


class DifferentialAttention(nn.Module):
    def __init__(self, config):
        """
        Initialize Differential Attention module.

        Args:
            config: Model configuration containing attention parameters
        """
        super().__init__()
        self.num_attention_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q1, Q2, K1, K2, V
        self.q_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Learnable parameters for λ
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_init = 0.8

    def compute_lambda(self):
        """Compute λ according to equation (2) in the paper."""
        return (torch.exp(self.lambda_q1 @ self.lambda_k1) -
                torch.exp(self.lambda_q2 @ self.lambda_k2) +
                self.lambda_init)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Project queries and keys twice, and values once
        q_combined = self.q_proj(hidden_states)
        k_combined = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split into Q1, Q2 and K1, K2
        q1, q2 = torch.chunk(q_combined, 2, dim=-1)
        k1, k2 = torch.chunk(k_combined, 2, dim=-1)

        # Split heads
        q1 = self.split_heads(q1)
        q2 = self.split_heads(q2)
        k1 = self.split_heads(k1)
        k2 = self.split_heads(k2)
        v = self.split_heads(v)

        # Compute attention scores
        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_scores1 = attn_scores1 + attention_mask
            attn_scores2 = attn_scores2 + attention_mask

        # Compute differential attention (equation 1)
        lambda_val = self.compute_lambda()
        attn_weights = (torch.softmax(attn_scores1, dim=-1) -
                        lambda_val * torch.softmax(attn_scores2, dim=-1))

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape output
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
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

            # Copy weights for the projections that are the same
            diff_attn.v_proj.weight.data = self.transformer.h[i].attn.c_attn.weight.data[:self.config.n_embd, :]
            diff_attn.out_proj.weight.data = self.transformer.h[i].attn.c_proj.weight.data

            # Initialize the split Q and K projections
            with torch.no_grad():
                # Split original Q and K weights for the dual attention
                orig_q_weights = self.transformer.h[i].attn.c_attn.weight.data[
                                 self.config.n_embd:2 * self.config.n_embd, :]
                orig_k_weights = self.transformer.h[i].attn.c_attn.weight.data[
                                 2 * self.config.n_embd:3 * self.config.n_embd, :]

                # Initialize the new dual Q and K projections
                diff_attn.q_proj.weight.data[:self.config.n_embd, :] = orig_q_weights
                diff_attn.q_proj.weight.data[self.config.n_embd:, :] = orig_q_weights

                diff_attn.k_proj.weight.data[:self.config.n_embd, :] = orig_k_weights
                diff_attn.k_proj.weight.data[self.config.n_embd:, :] = orig_k_weights

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
