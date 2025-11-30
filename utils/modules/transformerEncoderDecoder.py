import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# ------------------------------
# Transformer encoder/decoder blocks
# ------------------------------
class SimpleTransformerEncoder(nn.Module):
    """
    Wraps nn.TransformerEncoder but with a projection into model_dim and back.
    Takes flattened pixel embeddings (B, N, C) and returns encoded pixel embeddings (B, N, C).
    """
    def __init__(self, embed_dim=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # positional encoding will be added externally
        #self._reset_parameters()

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # src: (B, N, C) -> transformer expects (N, B, C)
        x = src.permute(1, 0, 2)
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2)
        return out

class SimpleTransformerDecoder(nn.Module):
    """
    Transformer decoder that takes queries (num_queries, embed_dim) and memory (pixel embeddings)
    and returns decoded query embeddings.
    """
    def __init__(self, embed_dim=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        #self._reset_parameters()

    def forward(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # tgt: (B, Q, C) -> (Q, B, C)
        # memory: (B, N, C) -> (N, B, C)
        tgt2 = tgt.permute(1, 0, 2)
        mem2 = memory.permute(1, 0, 2)
        out = self.decoder(tgt2, mem2, memory_key_padding_mask=memory_key_padding_mask)
        out = out.permute(1, 0, 2)  # (B, Q, C)
        return out

# ------------------------------
# Positional Encoding (2D sine)
# ------------------------------
class PositionEmbeddingSine(nn.Module):
    """
    2D sine positional encoding (like DETR).
    Input: mask_features with shape (B, C, H, W) — we compute positional enc for H x W and repeat across batch.
    Output: (B, H*W, embed_dim)
    """
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, mask: Tensor) -> Tensor:
        # mask is any image-like tensor with shape (B, C, H, W)
        B, C, H, W = mask.shape
        device = mask.device
        y_embed = torch.linspace(0, 1, steps=H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, steps=W, device=device).unsqueeze(0).repeat(H, 1)
        eps = 1e-6
        # create sine embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (self.num_pos_feats * 2))
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)  # (H, W, 2*num_pos_feats)
        pos = pos.view(1, H * W, 2 * self.num_pos_feats).repeat(B, 1, 1)
        return pos  # (B, N, C_pos) where C_pos = 2*num_pos_feats