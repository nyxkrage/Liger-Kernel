import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
    GPTJ_STYLE: tl.constexpr = False,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # q stride: (seq_len * num_q_heads * head_dim, num_q_heads * head_dim, head_dim, 1)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # k stride: (seq_len * num_kv_heads * head_dim, num_kv_heads * head_dim, head_dim, 1)

    # cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
    # stride: (seq_len * head_dim, head_dim, 1)
    pid = tl.program_id(0)

    # locate start address
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    batch_idx = pid // sl
    cos_row_idx = pid % sl
    cos = cos + tl.where(
        cos_bs == 1,
        cos_row_idx * cos_row_stride,
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
    )
    sin = sin + tl.where(
        cos_bs == 1,
        cos_row_idx * sin_row_stride,
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
    )

    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    if not GPTJ_STYLE:
        # Default Llama/GPT-NeoX style RoPE
        # left half of the head
        first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
        first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
        q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
        k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

        # right half of the head
        second_half_q_offsets = first_half_q_offsets + (hd // 2)
        second_half_k_offsets = first_half_k_offsets + (hd // 2)
        second_q_mask = first_q_mask
        second_k_mask = first_k_mask
        q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
        k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

        if not BACKWARD_PASS:
            # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
            new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
            tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
            new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
            tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

            new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
            tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
            new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
            tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
        else:
            # with some math, we can get:
            # dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
            new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
            tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
            new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
            tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

            new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
            tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
            new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
            tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:  # GPTJ_STYLE is True
        # GPT-J style RoPE: rotate adjacent pairs of features
        # (x_0, x_1), (x_2, x_3), ..., (x_{hd-2}, x_{hd-1})
        # cos_row and sin_row are loaded based on hd // 2.
        # Each element cos_row[i] and sin_row[i] corresponds to the pair (x_{2i}, x_{2i+1}).

        # Offsets for even features (x_0, x_2, ..., x_{hd-2})
        # tl.arange(0, pad_hd, 2) will generate [0, 2, ..., pad_hd-2]
        # These are then shaped to (1, pad_hd/2)
        q_head_offsets_even = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd, 2)[None, :]
        k_head_offsets_even = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd, 2)[None, :]
        
        # Mask for even features: compare with n_qh and actual hd // 2 valid features
        q_even_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
        k_even_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
        
        q_even = tl.load(q_ptr + q_head_offsets_even, mask=q_even_mask, other=0).to(sin_row.dtype)
        k_even = tl.load(k_ptr + k_head_offsets_even, mask=k_even_mask, other=0).to(sin_row.dtype)

        # Offsets for odd features (x_1, x_3, ..., x_{hd-1})
        # tl.arange(1, pad_hd, 2) will generate [1, 3, ..., pad_hd-1]
        q_head_offsets_odd = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(1, pad_hd, 2)[None, :]
        k_head_offsets_odd = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(1, pad_hd, 2)[None, :]
        
        # Mask for odd features: same logic as even features
        q_odd_mask = q_even_mask # Dimensions are the same: n_heads x (hd/2)
        k_odd_mask = k_even_mask # Dimensions are the same: n_heads x (hd/2)

        q_odd = tl.load(q_ptr + q_head_offsets_odd, mask=q_odd_mask, other=0).to(sin_row.dtype)
        k_odd = tl.load(k_ptr + k_head_offsets_odd, mask=k_odd_mask, other=0).to(sin_row.dtype)
        
        if not BACKWARD_PASS:
            # Forward pass:
            # new_x_even = x_even * cos_row - x_odd * sin_row
            # new_x_odd  = x_odd * cos_row + x_even * sin_row
            new_q_even = q_even * cos_row - q_odd * sin_row
            new_q_odd = q_odd * cos_row + q_even * sin_row
            tl.store(q_ptr + q_head_offsets_even, new_q_even, mask=q_even_mask)
            tl.store(q_ptr + q_head_offsets_odd, new_q_odd, mask=q_odd_mask)

            new_k_even = k_even * cos_row - k_odd * sin_row
            new_k_odd = k_odd * cos_row + k_even * sin_row
            tl.store(k_ptr + k_head_offsets_even, new_k_even, mask=k_even_mask)
            tl.store(k_ptr + k_head_offsets_odd, new_k_odd, mask=k_odd_mask)
        else:
            # Backward pass:
            # new_x_even = x_even * cos_row + x_odd * sin_row
            # new_x_odd  = x_odd * cos_row - x_even * sin_row
            new_q_even = q_even * cos_row + q_odd * sin_row
            new_q_odd = q_odd * cos_row - q_even * sin_row
            tl.store(q_ptr + q_head_offsets_even, new_q_even, mask=q_even_mask)
            tl.store(q_ptr + q_head_offsets_odd, new_q_odd, mask=q_odd_mask)

            new_k_even = k_even * cos_row + k_odd * sin_row
            new_k_odd = k_odd * cos_row - k_even * sin_row
            tl.store(k_ptr + k_head_offsets_even, new_k_even, mask=k_even_mask)
            tl.store(k_ptr + k_head_offsets_odd, new_k_odd, mask=k_odd_mask)


def rope_forward(q, k, cos, sin, gptj_style: bool = False):
    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_batch_size = cos.shape[0]

    _triton_rope[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=False,
    GPTJ_STYLE=gptj_style,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin, gptj_style: bool = False):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    cos_batch_size = cos.shape[0]
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure dq and dk are contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()

    # backward is similar to forward except swapping few ops
    _triton_rope[(n_row,)](
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=True,
    GPTJ_STYLE=gptj_style,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerRopeFunction(torch.autograd.Function):
    """
    Triton implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2

    This kernel also supports GPT-J style RoPE, where rotation is applied to adjacent pairs of features.
    This behavior is controlled by the `gptj_style` argument passed to the `forward` method.
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1, gptj_style: bool = False):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        gptj_style: bool, whether to use GPT-J style RoPE.
        """
        q_out, k_out, cos_out, sin_out = rope_forward(q, k, cos, sin, gptj_style=gptj_style)
        ctx.save_for_backward(cos_out, sin_out)
        ctx.gptj_style = gptj_style # Save for backward pass
        return q_out, k_out

    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """

        cos, sin = ctx.saved_tensors
        gptj_style = ctx.gptj_style

        dq_out, dk_out = rope_backward(dq, dk, cos, sin, gptj_style=gptj_style)
        # The forward signature is (ctx, q, k, cos, sin, position_ids, unsqueeze_dim, gptj_style)
        # So backward needs to return gradients for q, k, cos, sin, position_ids, unsqueeze_dim, gptj_style
        return dq_out, dk_out, None, None, None, None, None
