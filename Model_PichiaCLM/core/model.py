from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import PAD_IDX


class MultiTaskSeq2Seq(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int = 25,
        dna_vocab_size: int = 67,
        hidden_size_enc: int = 510,
        hidden_size_enc_aa: int = 510,
        embedding_size_enc: int = 42,
        embedding_size_dec: int = 224,
        dense_layer_size: int = 125,
        dense_layer_size_aa: int = 139,
        drop_rate: float = 0.0,
        drop_rate_aa: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc_emb = nn.Embedding(aa_vocab_size, embedding_size_enc, padding_idx=PAD_IDX)
        self.encoder = nn.GRU(
            embedding_size_enc,
            hidden_size_enc,
            batch_first=True,
            bidirectional=True,
        )
        self.dec_emb_cds = nn.Embedding(dna_vocab_size, embedding_size_dec, padding_idx=PAD_IDX)
        self.decoder_cds = nn.GRU(
            embedding_size_dec,
            2 * hidden_size_enc,
            batch_first=True,
        )
        self.decoder_aa = nn.GRU(
            embedding_size_enc,
            2 * hidden_size_enc_aa,
            batch_first=True,
        )
        self.inter_dense_cds = nn.Linear(4 * hidden_size_enc, dense_layer_size)
        self.inter_dense_aa = nn.Linear(
            2 * hidden_size_enc_aa + 2 * hidden_size_enc,
            dense_layer_size_aa,
        )
        self.dropout_cds = nn.Dropout(drop_rate)
        self.dropout_aa = nn.Dropout(drop_rate_aa)
        self.out_cds = nn.Linear(dense_layer_size, dna_vocab_size)
        self.out_aa = nn.Linear(dense_layer_size_aa, aa_vocab_size)

    def dot_product_attention(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        d_k = query.size(-1)
        scores = torch.bmm(query, value.transpose(1, 2)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, value)
