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

        # Single projection for Q, K, V (following GPT-2's architecture)
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Learnable parameter for λ
        self.lambda_param = nn.Parameter(torch.tensor(0.8))

        # Save config for later use
        self.config = config

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Project input to Q, K, V using single projection (GPT-2 style)
        qkv_combined = self.c_attn(hidden_states)  # [batch, seq, 3*hidden]

        # Split into Q, K, V
        q, k, v = qkv_combined.chunk(3, dim=-1)

        # Split Q and K for differential attention
        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)

        # Reshape for multi-head attention
        def reshape_for_attention(x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        # Reshape all tensors
        q1 = reshape_for_attention(q1)  # [batch, head, seq, head_dim]
        q2 = reshape_for_attention(q2)
        k1 = reshape_for_attention(k1)
        k2 = reshape_for_attention(k2)
        v = reshape_for_attention(v)

        # Handle layer past if provided
        if layer_past is not None:
            past_k, past_v = layer_past
            k1 = torch.cat([past_k[:, :, :, :self.head_dim], k1], dim=-2)
            k2 = torch.cat([past_k[:, :, :, self.head_dim:], k2], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Save current keys and values if using cache
        if use_cache:
            present = (torch.cat([k1, k2], dim=-1), v)
        else:
            present = None

        # Apply head mask if provided
        if head_mask is not None:
            q1 = q1 * head_mask.unsqueeze(-1).unsqueeze(-1)
            q2 = q2 * head_mask.unsqueeze(-1).unsqueeze(-1)

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
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_attn_output_shape = attn_output.size()[:-2] + (self.hidden_size,)
        attn_output = attn_output.view(new_attn_output_shape)
        attn_output = self.c_proj(attn_output)

        # Prepare outputs
        outputs = (attn_output,)
        if use_cache:
            outputs += (present,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)


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
        num_layers = len(self.transformer.h)

        # Calculate number of layers to replace
        num_replace = int(num_layers * differential_ratio)

        # Replace attention layers starting from the top
        for i in range(num_layers - num_replace, num_layers):
            # Create new differential attention layer with same config
            diff_attn = DifferentialAttention(self.transformer.h[i].attn.config)

            # Copy weights from original attention
            with torch.no_grad():
                # Copy the combined QKV weights
                diff_attn.c_attn.weight.data = self.transformer.h[i].attn.c_attn.weight.data.clone()
                diff_attn.c_attn.bias.data = self.transformer.h[i].attn.c_attn.bias.data.clone()

                # Copy the output projection
                diff_attn.c_proj.weight.data = self.transformer.h[i].attn.c_proj.weight.data.clone()
                diff_attn.c_proj.bias.data = self.transformer.h[i].attn.c_proj.bias.data.clone()

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
