import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.transformers.functional import liger_rope
from liger_kernel.transformers.rope import liger_rotary_pos_emb, LigerRotaryEmbedding # Added LigerRotaryEmbedding
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()

SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q/k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird shapes
        # HuggingFace llama/mistral source code doesn't support odd head dimension
        # so we don't test it here
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    expand_position_ids,
    atol,
    rtol,
):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)

    # validate forward pass
    hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin, pos_ids)
    tt_q, tt_k = liger_rotary_pos_emb(q2, k2, cos, sin)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # validate backward pass
    dq, dk = (
        torch.randn_like(hf_q, device=device),
        torch.randn_like(hf_k, device=device).to(dtype),
    )

    q1_grad, k1_grad = torch.autograd.grad((hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


# Reference implementation for GPT-J style RoPE
def gptj_rope_reference(x, cos_cached, sin_cached, head_dim):
    """
    Reference PyTorch implementation for GPT-J style Rotary Positional Embedding.

    Args:
        x (torch.Tensor): Input tensor (query or key) of shape (bsz, num_heads, seq_len, head_dim).
        cos_cached (torch.Tensor): Cosine cache, shape (bsz, seq_len, 1, head_dim) or (1, seq_len, 1, head_dim).
                                   The last dimension contains duplicated halves.
        sin_cached (torch.Tensor): Sine cache, shape (bsz, seq_len, 1, head_dim) or (1, seq_len, 1, head_dim).
                                   The last dimension contains duplicated halves.
        head_dim (int): The head dimension.

    Returns:
        torch.Tensor: Output tensor with RoPE applied, same shape as x.
    """
    # x: (bsz, num_heads, seq_len, head_dim)
    # cos_cached, sin_cached: e.g., (bsz, seq_len, 1, head_dim)

    # Select the first half for actual rotation values
    # These will be (bsz, seq_len, 1, head_dim // 2) or (1, seq_len, 1, head_dim // 2)
    cos_for_rotation_full_slice = cos_cached[..., :head_dim // 2]
    sin_for_rotation_full_slice = sin_cached[..., :head_dim // 2]

    # Reshape for broadcasting with x_even/x_odd
    # Target shape for cos/sin: (bsz or 1, 1, seq_len, head_dim // 2)
    # Current shape of slice:   (bsz or 1, seq_len, 1, head_dim // 2)
    
    # Squeeze the dim of size 1 at index -2 (original head_dim dim split into two)
    cos_for_rotation = cos_for_rotation_full_slice.squeeze(-2) # (bsz or 1, seq_len, head_dim // 2)
    sin_for_rotation = sin_for_rotation_full_slice.squeeze(-2) # (bsz or 1, seq_len, head_dim // 2)

    # Unsqueeze to add num_heads dimension for broadcasting
    cos_for_rotation = cos_for_rotation.unsqueeze(1) # (bsz or 1, 1, seq_len, head_dim // 2)
    sin_for_rotation = sin_for_rotation.unsqueeze(1) # (bsz or 1, 1, seq_len, head_dim // 2)


    x_even = x[..., ::2]  # (bsz, num_heads, seq_len, head_dim // 2)
    x_odd = x[..., 1::2]   # (bsz, num_heads, seq_len, head_dim // 2)

    # x_rotated_even = x_even * cos_for_rotation - x_odd * sin_for_rotation
    # x_rotated_odd  = x_odd * cos_for_rotation + x_even * sin_for_rotation (as per prompt)
    x_rotated_even = x_even * cos_for_rotation - x_odd * sin_for_rotation
    x_rotated_odd = x_even * sin_for_rotation + x_odd * cos_for_rotation # Typical GPT-J / EleutherAI impl

    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rotated_even
    x_out[..., 1::2] = x_rotated_odd
    return x_out


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 128),
        (1, 64, 16, 16, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1, # Looser tolerance for bfloat16
            1e-2, # Looser tolerance for bfloat16
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_gptj_rope_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    expand_position_ids,
    atol,
    rtol,
):
    # Use HuggingFace's LlamaRotaryEmbedding to generate cos/sin caches
    # This is a well-tested component and ensures the inputs to our RoPE are standard.
    hf_rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    _tensor_q_orig = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)
    _tensor_k_orig = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    # Tensors for Liger kernel
    q_liger = _tensor_q_orig.clone().requires_grad_(True)
    k_liger = _tensor_k_orig.clone().requires_grad_(True)

    # Tensors for reference implementation
    q_ref_input = _tensor_q_orig.clone() # No grad needed for ref input, grad checked on liger's input
    k_ref_input = _tensor_k_orig.clone()

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    
    # cos/sin from HF LlamaRotaryEmbedding. These are typically (bsz, seq_len, 1, head_dim) or (1, seq_len, 1, head_dim)
    # The LigerRotaryEmbedding class expects cos/sin in this format (or compatible)
    # These caches have the duplicated halves in the last dimension.
    cos_cache, sin_cache = hf_rotary_emb(q_liger, pos_ids) # Pass q_liger to match device and context

    # Instantiate LigerRotaryEmbedding with GPT-J style
    liger_gptj_rope = LigerRotaryEmbedding(rope_type="gptj")

    # Forward pass with Liger's RoPE
    # LigerRotaryEmbedding.forward takes (q, k, cos, sin, position_ids), position_ids is optional.
    # The cos and sin here are already prepared for the sequence.
    liger_q_out, liger_k_out = liger_gptj_rope.forward(q_liger, k_liger, cos_cache, sin_cache, position_ids=None)

    # Forward pass with reference GPT-J RoPE
    ref_q_out = gptj_rope_reference(q_ref_input, cos_cache, sin_cache, head_dim)
    ref_k_out = gptj_rope_reference(k_ref_input, cos_cache, sin_cache, head_dim)

    # Validate forward pass
    assert torch.allclose(liger_q_out, ref_q_out, atol=atol, rtol=rtol), "Mismatch in Q tensor forward pass"
    assert torch.allclose(liger_k_out, ref_k_out, atol=atol, rtol=rtol), "Mismatch in K tensor forward pass"

    # Validate backward pass (check if gradients are computed)
    dq_dummy = torch.randn_like(liger_q_out)
    dk_dummy = torch.randn_like(liger_k_out)
    
    # Autograd for Liger output
    torch.autograd.grad((liger_q_out, liger_k_out), (q_liger, k_liger), (dq_dummy, dk_dummy))

    assert q_liger.grad is not None, "Gradient not computed for Q tensor"
    assert k_liger.grad is not None, "Gradient not computed for K tensor"


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 2, 2, 2, 8),
        (1, 2, 1, 2, 8),
        # weird shapes
        (9, 7, 41, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_functional_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    expand_position_ids,
    dtype,
    atol,
    rtol,
):
    _q = torch.randn((bsz, num_q_heads, seq_len, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)

    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)

    functional_q, functional_k = liger_rope(q=q1, k=k1, cos=cos, sin=sin)
    class_q, class_k = LigerRopeFunction.apply(q2, k2, cos, sin)

    assert torch.allclose(functional_q, class_q, atol=atol, rtol=rtol)
    assert torch.allclose(functional_k, class_k, atol=atol, rtol=rtol)

    dq, dk = torch.randn_like(functional_q), torch.randn_like(functional_k)

    dq1, dk1 = dq.clone(), dk.clone()
    dq2, dk2 = dq.clone(), dk.clone()

    q1_grad, k1_grad = torch.autograd.grad(
        (functional_q, functional_k),
        (q1, k1),
        (dq1, dk1),
        allow_unused=True,
    )

    q2_grad, k2_grad = torch.autograd.grad(
        (class_q, class_k),
        (q2, k2),
        (dq2, dk2),
        allow_unused=True,
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)
