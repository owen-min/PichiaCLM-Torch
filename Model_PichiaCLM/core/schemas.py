from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    amino_acids: str
    cds: str
    codon_ids: list[int]
    device: str
