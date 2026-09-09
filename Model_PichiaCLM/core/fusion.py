from __future__ import annotations

from dataclasses import asdict, dataclass

from .analysis import SequenceAnalysisReport, analyze_cds
from .biology import normalize_amino_acid_sequence
from .predictor import PichiaCLMPredictor
from .schemas import PredictionResult


@dataclass(frozen=True)
class CleavageWindow:
    amino_acid_start: int
    amino_acid_end: int
    cds_start: int
    cds_end: int
    amino_acids: str
    cds: str


@dataclass(frozen=True)
class FusionPrediction:
    mode: str
    prediction: PredictionResult
    analysis: SequenceAnalysisReport
    cleavage_window: CleavageWindow


@dataclass(frozen=True)
class FusionComparison:
    signal_peptide: str
    mature_protein: str
    fused_amino_acids: str
    whole_sequence: FusionPrediction
    segmented: FusionPrediction
    cds_are_identical: bool


def compare_signal_fusion(
    predictor: PichiaCLMPredictor,
    signal_peptide: str,
    mature_protein: str,
    allow_unknown: bool = False,
    cleavage_flank_aa: int = 5,
) -> FusionComparison:
    signal = normalize_amino_acid_sequence(signal_peptide)
    mature = normalize_amino_acid_sequence(mature_protein)
    if not signal:
        raise ValueError("Signal peptide sequence must not be empty.")
    if not mature:
        raise ValueError("Mature protein sequence must not be empty.")

    fused = signal + mature
    whole_prediction = predictor.predict(fused, allow_unknown=allow_unknown)
    signal_prediction = predictor.predict(signal, allow_unknown=allow_unknown)
    mature_prediction = predictor.predict(mature, allow_unknown=allow_unknown)
    segmented_prediction = PredictionResult(
        amino_acids=fused,
        cds=signal_prediction.cds + mature_prediction.cds,
        codon_ids=signal_prediction.codon_ids + mature_prediction.codon_ids,
        device=whole_prediction.device,
    )

    return FusionComparison(
        signal_peptide=signal,
        mature_protein=mature,
        fused_amino_acids=fused,
        whole_sequence=FusionPrediction(
            mode="whole",
            prediction=whole_prediction,
            analysis=analyze_cds(whole_prediction.cds, amino_acids=fused),
            cleavage_window=_cleavage_window(fused, whole_prediction.cds, len(signal), cleavage_flank_aa),
        ),
        segmented=FusionPrediction(
            mode="segmented",
            prediction=segmented_prediction,
            analysis=analyze_cds(segmented_prediction.cds, amino_acids=fused),
            cleavage_window=_cleavage_window(fused, segmented_prediction.cds, len(signal), cleavage_flank_aa),
        ),
        cds_are_identical=whole_prediction.cds == segmented_prediction.cds,
    )


def fusion_to_dict(comparison: FusionComparison) -> dict[str, object]:
    return asdict(comparison)


def _cleavage_window(
    fused_amino_acids: str,
    cds: str,
    signal_length: int,
    flank_aa: int,
) -> CleavageWindow:
    aa_start = max(1, signal_length - flank_aa + 1)
    aa_end = min(len(fused_amino_acids), signal_length + flank_aa)
    cds_start = (aa_start - 1) * 3 + 1
    cds_end = aa_end * 3
    return CleavageWindow(
        amino_acid_start=aa_start,
        amino_acid_end=aa_end,
        cds_start=cds_start,
        cds_end=cds_end,
        amino_acids=fused_amino_acids[aa_start - 1 : aa_end],
        cds=cds[cds_start - 1 : cds_end],
    )
