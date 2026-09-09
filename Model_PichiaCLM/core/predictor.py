from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .config import DEFAULT_WEIGHTS_PATH
from .model import MultiTaskSeq2Seq
from .schemas import PredictionResult
from .vocab import AA_EOS_IDX, CDS_SOS_IDX, build_vocabularies, normalize_amino_acids


class PichiaCLMPredictor:
    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.aa_vocab, self.idx_to_codon, self.aa_id_to_codon_ids = build_vocabularies()
        self.valid_amino_acids = frozenset(self.aa_id_to_codon_ids)

        self.model = MultiTaskSeq2Seq().to(self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, amino_acids: str, allow_unknown: bool = False) -> PredictionResult:
        normalized = normalize_amino_acids(amino_acids)
        aa_indices = self._encode_amino_acids(normalized, allow_unknown=allow_unknown)
        aa_tensor = torch.tensor([aa_indices + [AA_EOS_IDX]], dtype=torch.long, device=self.device)
        codon_ids = self._translate_aa_to_cds(aa_tensor)
        cds = "".join(self.idx_to_codon.get(idx, "") for idx in codon_ids)
        return PredictionResult(
            amino_acids=normalized,
            cds=cds,
            codon_ids=codon_ids,
            device=str(self.device),
        )

    def _encode_amino_acids(self, amino_acids: str, allow_unknown: bool) -> list[int]:
        encoded: list[int] = []
        invalid_chars: list[str] = []
        for char in amino_acids:
            aa_id = self.aa_vocab.get(char)
            if aa_id is None:
                invalid_chars.append(char)
                continue
            if not allow_unknown and aa_id not in self.valid_amino_acids:
                invalid_chars.append(char)
                continue
            encoded.append(aa_id)

        if invalid_chars:
            unique = ", ".join(sorted(set(invalid_chars)))
            raise ValueError(f"Unsupported amino acid character(s): {unique}")
        return encoded

    def _translate_aa_to_cds(self, aa_tensor: torch.Tensor) -> list[int]:
        predicted_cds_indices: list[int] = []
        with torch.no_grad():
            enc_emb_out = self.model.enc_emb(aa_tensor)
            enc_seq, enc_hidden = self.model.encoder(enc_emb_out)
            hidden_state = torch.cat([enc_hidden[0], enc_hidden[1]], dim=-1).unsqueeze(0)
            decoder_input = torch.tensor([[CDS_SOS_IDX]], dtype=torch.long, device=self.device)

            for counter in range(aa_tensor.size(1)):
                current_aa_id = aa_tensor[0, counter].item()
                if current_aa_id == AA_EOS_IDX:
                    break

                dec_emb_cds_out = self.model.dec_emb_cds(decoder_input)
                dec_seq_cds, hidden_state = self.model.decoder_cds(
                    dec_emb_cds_out,
                    hidden_state,
                )
                attn_out_cds = self.model.dot_product_attention(dec_seq_cds, enc_seq)
                concat_cds = torch.cat([dec_seq_cds, attn_out_cds], dim=-1)
                inter_cds = torch.tanh(self.model.inter_dense_cds(concat_cds))
                current_logits = self.model.out_cds(inter_cds)[:, -1, :]

                valid_codon_ids = self.aa_id_to_codon_ids.get(current_aa_id, [])
                if valid_codon_ids:
                    mask = torch.full_like(current_logits, float("-inf"))
                    for codon_id in valid_codon_ids:
                        mask[0, codon_id] = 0.0
                    current_logits = current_logits + mask

                next_token = torch.argmax(current_logits, dim=-1).item()
                predicted_cds_indices.append(next_token)
                decoder_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        return predicted_cds_indices


def batch_predict(
    predictor: PichiaCLMPredictor,
    sequences: Iterable[str],
    allow_unknown: bool = False,
) -> list[PredictionResult]:
    return [predictor.predict(sequence, allow_unknown=allow_unknown) for sequence in sequences]
