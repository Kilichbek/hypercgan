"""Word-level attention-based weight modulation (HyperCGAN, Sec. 3)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

#----------------------------------------------------------------------------

def tensor_modulation_word(weight: Tensor, factors: Tensor, tokens: Tensor=None, activation: str='lrelu') -> Tensor:
    """
    Builds a multiplicative modulation tensor M for a [c_out, c_in] weight from word embeddings.

    Every word predicts a rank-1 modulation `u_i v_i^T`. The (already style-modulated)
    weight W attends over these word-level modulations, and the attention-weighted
    sum becomes the modulation tensor.

    Args:
        weight:     [batch_size, c_out, c_in] -- current (style-modulated) weight W.
        factors:    [batch_size, num_words, c_out + c_in] -- hypernetwork outputs (u_i, v_i) per word.
        tokens:     [batch_size, num_words] -- CLIP token ids; padding tokens (id 0) are masked out.
        activation: non-linearity applied to the normalized modulation ('lrelu', 'tanh', 'sigmoid', 'linear').

    Returns:
        M: [batch_size, c_out, c_in] -- multiplicative modulation, centered around 1.
    """
    b, c_out, c_in = weight.shape
    n_words = factors.shape[1]

    # Rank-1 modulation per word.
    M = factors[:, :, :c_out].view(b, n_words, c_out, 1) \
        * factors[:, :, c_out:].view(b, n_words, 1, c_in) # [b, num_words, c_out, c_in]

    # Attention of the weight over the words.
    M = M.view(b, n_words, -1)                                          # [b, num_words, c_out * c_in]
    weight = weight.view(b, 1, -1)                                      # [b, 1, c_out * c_in]
    score = torch.bmm(weight, M.transpose(1, 2)) / np.sqrt(c_out * c_in) # [b, 1, num_words]
    if tokens is not None:
        mask = (tokens == 0).unsqueeze(1).expand(b, 1, n_words)         # [b, 1, num_words]
        score = score.masked_fill(mask, float('-inf'))
    attn = F.softmax(score, -1)                                         # [b, 1, num_words]
    M = torch.bmm(attn, M).view(b, c_out, c_in)                         # [b, c_out, c_in]

    M = M / M.norm(float('inf'), dim=[1, 2], keepdim=True)
    if activation == 'tanh':
        M = torch.tanh(M)
    elif activation == 'lrelu':
        M = F.leaky_relu(M, 0.2)
    elif activation == 'sigmoid':
        M = torch.sigmoid(M) - 0.5
    elif activation != 'linear':
        raise ValueError(f'Unknown activation function: {activation}')

    return M + 1.0

#----------------------------------------------------------------------------
