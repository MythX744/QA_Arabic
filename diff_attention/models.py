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
        """
        Initialize GPT2 model for Question Answering with differential attention.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Linear(config.n_embd, 2)  # For start/end position prediction
        self.init_weights()

    def replace_attention_layers(self, ratio=0.3):
        """Replace portion of attention layers with differential attention."""
        num_layers = len(self.transformer.h)
        num_to_replace = int(num_layers * ratio)

        for i in range(0, num_layers, num_layers // num_to_replace):
            if i < num_layers:
                self.transformer.h[i].attn = DifferentialAttention(self.config)
                print(f"Replaced attention layer {i} with differential attention")

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
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

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


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
