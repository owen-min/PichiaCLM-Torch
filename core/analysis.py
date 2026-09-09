from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .biology import check_translation
from .restriction import RestrictionSite, parse_custom_sites, scan_restriction_sites


GLOBAL_GC_MIN = 35.0
GLOBAL_GC_MAX = 65.0
LOCAL_GC_WINDOW = 30
LOCAL_GC_MIN = 25.0
LOCAL_GC_MAX = 75.0
HOMOPOLYMER_MIN_LENGTH = 6
RARE_CODON_FRACTION = 0.10
RARE_CODON_RUN_MIN_LENGTH = 2

TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "Training" / "AllData"


CODON_TO_AA = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

AA_TO_CODONS: dict[str, list[str]] = defaultdict(list)
for _codon, _aa in CODON_TO_AA.items():
    AA_TO_CODONS[_aa].append(_codon)
AA_TO_CODONS = dict(AA_TO_CODONS)


# Kazusa Codon Usage Database: Pichia pastoris [gbpln], taxonomy id 4922,
# 137 CDS / 81301 codons, standard genetic code.
PUBLIC_PICHIA_PASTORIS_FRACTIONS = {
    "TTT": 0.54,
    "TTC": 0.46,
    "TTA": 0.16,
    "TTG": 0.33,
    "TCT": 0.29,
    "TCC": 0.20,
    "TCA": 0.18,
    "TCG": 0.09,
    "TAT": 0.47,
    "TAC": 0.53,
    "TAA": 0.51,
    "TAG": 0.29,
    "TGT": 0.64,
    "TGC": 0.36,
    "TGA": 0.20,
    "TGG": 1.00,
    "CTT": 0.16,
    "CTC": 0.08,
    "CTA": 0.11,
    "CTG": 0.16,
    "CCT": 0.35,
    "CCC": 0.15,
    "CCA": 0.42,
    "CCG": 0.09,
    "CAT": 0.57,
    "CAC": 0.43,
    "CAA": 0.61,
    "CAG": 0.39,
    "CGT": 0.17,
    "CGC": 0.05,
    "CGA": 0.10,
    "CGG": 0.05,
    "ATT": 0.50,
    "ATC": 0.31,
    "ATA": 0.18,
    "ATG": 1.00,
    "ACT": 0.40,
    "ACC": 0.26,
    "ACA": 0.24,
    "ACG": 0.11,
    "AAT": 0.48,
    "AAC": 0.52,
    "AAA": 0.47,
    "AAG": 0.53,
    "AGT": 0.15,
    "AGC": 0.09,
    "AGA": 0.48,
    "AGG": 0.16,
    "GTT": 0.42,
    "GTC": 0.23,
    "GTA": 0.15,
    "GTG": 0.19,
    "GCT": 0.45,
    "GCC": 0.26,
    "GCA": 0.23,
    "GCG": 0.06,
    "GAT": 0.58,
    "GAC": 0.42,
    "GAA": 0.56,
    "GAG": 0.44,
    "GGT": 0.44,
    "GGC": 0.14,
    "GGA": 0.33,
    "GGG": 0.10,
}


@dataclass(frozen=True)
class LocalGCWindow:
    start: int
    end: int
    gc_percent: float


@dataclass(frozen=True)
class HomopolymerRun:
    base: str
    start: int
    end: int
    length: int


@dataclass(frozen=True)
class RareCodonRun:
    reference: str
    start_codon: int
    end_codon: int
    codons: list[str]


@dataclass(frozen=True)
class TandemRepeat:
    sequence: str
    start: int
    end: int
    copies: int


@dataclass(frozen=True)
class RepeatedKmer:
    sequence: str
    count: int
    positions: list[int]


@dataclass(frozen=True)
class MotifHit:
    motif: str
    start: int
    end: int


@dataclass(frozen=True)
class CodonUsageRow:
    codon: str
    amino_acid: str
    count: int
    sequence_fraction: float
    training_fraction: float | None
    public_fraction: float | None


@dataclass(frozen=True)
class CAIComparison:
    training: float | None
    public: float | None
    delta: float | None


@dataclass(frozen=True)
class SequenceAnalysisReport:
    cds_length: int
    codon_count: int
    valid_dna: bool
    invalid_bases: list[str]
    length_multiple_of_three: bool
    translated_amino_acids: str
    translation_matches_input: bool | None
    internal_stop_codons: list[int]
    sequence_warnings: list[str]
    gc_percent: float
    gc_status: str
    local_gc_window: int
    local_gc_outliers: list[LocalGCWindow]
    cai: CAIComparison
    rare_codon_runs: list[RareCodonRun]
    homopolymers: list[HomopolymerRun]
    tandem_repeats: list[TandemRepeat]
    repeated_kmers: list[RepeatedKmer]
    motif_hits: list[MotifHit]
    restriction_sites: list[RestrictionSite]
    codon_usage: list[CodonUsageRow]
    training_reference_codon_count: int
    public_reference: str


def normalize_dna(sequence: str) -> str:
    return "".join(sequence.split()).upper().replace("U", "T")


def normalize_motifs(motifs: Iterable[str] | None) -> list[str]:
    if motifs is None:
        return []
    normalized = []
    for motif in motifs:
        clean = normalize_dna(motif)
        if clean:
            normalized.append(clean)
    return normalized


def split_codons(cds: str) -> list[str]:
    return [cds[index : index + 3] for index in range(0, len(cds) - 2, 3)]


def translate_cds(cds: str) -> str:
    return "".join(CODON_TO_AA.get(codon, "X") for codon in split_codons(cds))


def gc_percent(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc_count = sum(1 for base in sequence if base in {"G", "C"})
    return round(gc_count / len(sequence) * 100, 2)


def scan_local_gc(cds: str, window: int = LOCAL_GC_WINDOW) -> list[LocalGCWindow]:
    if len(cds) < window:
        return []
    outliers = []
    for index in range(0, len(cds) - window + 1):
        value = gc_percent(cds[index : index + window])
        if value < LOCAL_GC_MIN or value > LOCAL_GC_MAX:
            outliers.append(LocalGCWindow(start=index + 1, end=index + window, gc_percent=value))
    return outliers


def build_fraction_table(codon_counts: Counter[str]) -> dict[str, float]:
    aa_totals: Counter[str] = Counter()
    for codon, count in codon_counts.items():
        aa = CODON_TO_AA.get(codon)
        if aa:
            aa_totals[aa] += count

    fractions = {}
    for codon in CODON_TO_AA:
        aa = CODON_TO_AA[codon]
        total = aa_totals[aa]
        fractions[codon] = codon_counts[codon] / total if total else 0.0
    return fractions


@lru_cache(maxsize=1)
def load_training_codon_reference() -> tuple[dict[str, float], int]:
    codon_counts: Counter[str] = Counter()
    for csv_path in sorted(TRAINING_DATA_DIR.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                cds = normalize_dna(row[-1])
                if not cds or any(base not in {"A", "T", "G", "C"} for base in cds):
                    continue
                for codon in split_codons(cds):
                    if codon in CODON_TO_AA:
                        codon_counts[codon] += 1
    return build_fraction_table(codon_counts), sum(codon_counts.values())


def cai(codons: list[str], reference_fractions: dict[str, float]) -> float | None:
    weights = []
    for codon in codons:
        aa = CODON_TO_AA.get(codon)
        if aa is None or aa == "*":
            continue
        synonymous = AA_TO_CODONS[aa]
        max_fraction = max(reference_fractions.get(item, 0.0) for item in synonymous)
        fraction = reference_fractions.get(codon, 0.0)
        if max_fraction <= 0 or fraction <= 0:
            continue
        weights.append(fraction / max_fraction)

    if not weights:
        return None
    return round(math.exp(sum(math.log(weight) for weight in weights) / len(weights)), 4)


def sequence_codon_usage(
    codons: list[str],
    training_fractions: dict[str, float],
    public_fractions: dict[str, float],
) -> list[CodonUsageRow]:
    counts: Counter[str] = Counter(codon for codon in codons if codon in CODON_TO_AA)
    aa_totals: Counter[str] = Counter(CODON_TO_AA[codon] for codon in counts for _ in range(counts[codon]))
    rows = []
    for codon in sorted(CODON_TO_AA):
        aa = CODON_TO_AA[codon]
        total = aa_totals[aa]
        rows.append(
            CodonUsageRow(
                codon=codon,
                amino_acid=aa,
                count=counts[codon],
                sequence_fraction=round(counts[codon] / total, 4) if total else 0.0,
                training_fraction=round(training_fractions.get(codon, 0.0), 4),
                public_fraction=round(public_fractions.get(codon, 0.0), 4),
            )
        )
    return rows


def find_rare_codon_runs(
    codons: list[str],
    reference_name: str,
    reference_fractions: dict[str, float],
) -> list[RareCodonRun]:
    runs = []
    current: list[tuple[int, str]] = []
    for index, codon in enumerate(codons, start=1):
        aa = CODON_TO_AA.get(codon)
        is_rare = (
            aa is not None
            and aa != "*"
            and len(AA_TO_CODONS[aa]) > 1
            and reference_fractions.get(codon, 0.0) < RARE_CODON_FRACTION
        )
        if is_rare:
            current.append((index, codon))
            continue
        if len(current) >= RARE_CODON_RUN_MIN_LENGTH:
            runs.append(
                RareCodonRun(
                    reference=reference_name,
                    start_codon=current[0][0],
                    end_codon=current[-1][0],
                    codons=[codon for _, codon in current],
                )
            )
        current = []

    if len(current) >= RARE_CODON_RUN_MIN_LENGTH:
        runs.append(
            RareCodonRun(
                reference=reference_name,
                start_codon=current[0][0],
                end_codon=current[-1][0],
                codons=[codon for _, codon in current],
            )
        )
    return runs


def find_homopolymers(cds: str) -> list[HomopolymerRun]:
    runs = []
    for match in re.finditer(r"([ATGC])\1{%d,}" % (HOMOPOLYMER_MIN_LENGTH - 1), cds):
        runs.append(
            HomopolymerRun(
                base=match.group(1),
                start=match.start() + 1,
                end=match.end(),
                length=match.end() - match.start(),
            )
        )
    return runs


def find_tandem_repeats(cds: str, min_unit: int = 2, max_unit: int = 12, min_copies: int = 3) -> list[TandemRepeat]:
    repeats = []
    seen: set[tuple[int, int]] = set()
    for unit_size in range(min_unit, max_unit + 1):
        for start in range(0, len(cds) - unit_size * min_copies + 1):
            unit = cds[start : start + unit_size]
            copies = 1
            cursor = start + unit_size
            while cds[cursor : cursor + unit_size] == unit:
                copies += 1
                cursor += unit_size
            if copies >= min_copies and (start, cursor) not in seen:
                seen.add((start, cursor))
                repeats.append(
                    TandemRepeat(
                        sequence=unit,
                        start=start + 1,
                        end=cursor,
                        copies=copies,
                    )
                )
    return sorted(repeats, key=lambda item: (item.start, -(item.end - item.start)))[:50]


def find_repeated_kmers(cds: str, kmer_size: int = 12, min_count: int = 3) -> list[RepeatedKmer]:
    positions: dict[str, list[int]] = defaultdict(list)
    if len(cds) < kmer_size:
        return []
    for index in range(0, len(cds) - kmer_size + 1):
        positions[cds[index : index + kmer_size]].append(index + 1)
    repeats = [
        RepeatedKmer(sequence=kmer, count=len(items), positions=items[:10])
        for kmer, items in positions.items()
        if len(items) >= min_count
    ]
    return sorted(repeats, key=lambda item: (-item.count, item.sequence))[:50]


def find_motifs(cds: str, motifs: Iterable[str] | None) -> list[MotifHit]:
    hits = []
    for motif in normalize_motifs(motifs):
        start = 0
        while True:
            index = cds.find(motif, start)
            if index == -1:
                break
            hits.append(MotifHit(motif=motif, start=index + 1, end=index + len(motif)))
            start = index + 1
    return hits


def analyze_cds(
    cds: str,
    amino_acids: str | None = None,
    motifs: Iterable[str] | None = None,
    include_restriction_sites: bool = True,
    custom_restriction_sites: Iterable[str] | None = None,
) -> SequenceAnalysisReport:
    normalized_cds = normalize_dna(cds)
    translation_check = check_translation(normalized_cds, amino_acids=amino_acids)
    codons = split_codons(normalized_cds)
    training_fractions, training_count = load_training_codon_reference()
    public_fractions = PUBLIC_PICHIA_PASTORIS_FRACTIONS

    training_cai = cai(codons, training_fractions)
    public_cai = cai(codons, public_fractions)
    delta = None
    if training_cai is not None and public_cai is not None:
        delta = round(training_cai - public_cai, 4)

    global_gc = gc_percent(normalized_cds)
    if global_gc < GLOBAL_GC_MIN:
        gc_status = "low"
    elif global_gc > GLOBAL_GC_MAX:
        gc_status = "high"
    else:
        gc_status = "ok"

    restriction_sites = []
    if include_restriction_sites:
        restriction_sites = scan_restriction_sites(
            normalized_cds,
            include_defaults=True,
            custom_sites=parse_custom_sites(custom_restriction_sites),
        )

    return SequenceAnalysisReport(
        cds_length=translation_check.cds_length,
        codon_count=translation_check.codon_count,
        valid_dna=translation_check.valid_dna,
        invalid_bases=translation_check.invalid_bases,
        length_multiple_of_three=translation_check.length_multiple_of_three,
        translated_amino_acids=translation_check.translated_amino_acids,
        translation_matches_input=translation_check.translation_matches_input,
        internal_stop_codons=translation_check.internal_stop_codons,
        sequence_warnings=translation_check.sequence_warnings,
        gc_percent=global_gc,
        gc_status=gc_status,
        local_gc_window=LOCAL_GC_WINDOW,
        local_gc_outliers=scan_local_gc(normalized_cds),
        cai=CAIComparison(training=training_cai, public=public_cai, delta=delta),
        rare_codon_runs=find_rare_codon_runs(codons, "training", training_fractions)
        + find_rare_codon_runs(codons, "public", public_fractions),
        homopolymers=find_homopolymers(normalized_cds),
        tandem_repeats=find_tandem_repeats(normalized_cds),
        repeated_kmers=find_repeated_kmers(normalized_cds),
        motif_hits=find_motifs(normalized_cds, motifs),
        restriction_sites=restriction_sites,
        codon_usage=sequence_codon_usage(codons, training_fractions, public_fractions),
        training_reference_codon_count=training_count,
        public_reference="Kazusa Pichia pastoris taxon 4922, 137 CDS / 81301 codons",
    )
