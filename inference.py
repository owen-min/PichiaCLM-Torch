"""Backward-compatible imports for the old inference module path."""

from .core.config import DEFAULT_WEIGHTS_PATH
from .core.analysis import SequenceAnalysisReport, analyze_cds
from .core.biology import TranslationCheck, check_translation, translate_cds
from .core.fasta import FastaRecord, format_fasta, parse_fasta
from .core.fusion import FusionComparison, compare_signal_fusion
from .core.model import MultiTaskSeq2Seq
from .core.postprocess import PostprocessResult, conservative_postprocess
from .core.predictor import PichiaCLMPredictor, batch_predict
from .core.restriction import RestrictionSite, scan_restriction_sites
from .core.schemas import PredictionResult
from .core.vocab import (
    AA_EOS_IDX,
    AA_UNK_IDX,
    CDS_SOS_IDX,
    PAD_IDX,
    build_vocabularies,
    normalize_amino_acids,
)

__all__ = [
    "AA_EOS_IDX",
    "AA_UNK_IDX",
    "CDS_SOS_IDX",
    "DEFAULT_WEIGHTS_PATH",
    "FastaRecord",
    "FusionComparison",
    "MultiTaskSeq2Seq",
    "PAD_IDX",
    "PichiaCLMPredictor",
    "PostprocessResult",
    "PredictionResult",
    "RestrictionSite",
    "SequenceAnalysisReport",
    "TranslationCheck",
    "analyze_cds",
    "batch_predict",
    "build_vocabularies",
    "check_translation",
    "compare_signal_fusion",
    "conservative_postprocess",
    "format_fasta",
    "normalize_amino_acids",
    "parse_fasta",
    "scan_restriction_sites",
    "translate_cds",
]
