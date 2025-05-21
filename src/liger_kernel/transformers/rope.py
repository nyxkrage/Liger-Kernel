from liger_kernel.ops.rope import LigerRopeFunction


import torch # Added import for torch.nn.Module

class LigerRotaryEmbedding(torch.nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.

    This module applies RoPE to query and key tensors. It can be configured
    to use either the standard Llama-style RoPE or the GPT-J style RoPE.
    """
    def __init__(self, unsqueeze_dim: int = 1, rope_type: str = "llama"):
        """
        Initializes the LigerRotaryEmbedding module.

        Args:
            unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.
                                           This corresponds to the `unsqueeze_dim` argument in
                                           `LigerRopeFunction.apply`.
            rope_type (str, optional): The type of RoPE to apply.
                                       Supported values are "llama" (default) and "gptj".
        """
        super().__init__()
        self.unsqueeze_dim = unsqueeze_dim
        self.rope_type = rope_type

    def forward(self, q, k, cos, sin, position_ids=None):
        """
        Applies Rotary Positional Embedding (RoPE) operation to query and key states.

        Args:
            q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
            k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
            cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
            sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
            position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
        """
        gptj_style = self.rope_type == "gptj"
        return LigerRopeFunction.apply(q, k, cos, sin, position_ids, self.unsqueeze_dim, gptj_style)


def liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, gptj_style: bool = False): # Added gptj_style for consistency
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """
    # Note: The original liger_rotary_pos_emb function did not have gptj_style.
    # For direct calls to this function, gptj_style would need to be passed explicitly.
    # The LigerRotaryEmbedding class now handles this based on its rope_type.
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim, gptj_style)
