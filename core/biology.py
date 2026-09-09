from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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

AA_TO_CODONS: dict[str, list[str]] = {}
for codon, aa in CODON_TO_AA.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

DNA_BASES = frozenset({"A", "T", "G", "C"})
SHORT_AA_LENGTH = 5
LONG_AA_LENGTH = 3000


@dataclass(frozen=True)
class TranslationCheck:
    cds_length: int
    codon_count: int
    valid_dna: bool
    invalid_bases: list[str]
    length_multiple_of_three: bool
    translated_amino_acids: str
    translation_matches_input: bool | None
    internal_stop_codons: list[int]
    sequence_warnings: list[str]


def normalize_dna(sequence: str) -> str:
    return "".join(sequence.split()).upper().replace("U", "T")


def normalize_amino_acid_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


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
    normalized = normalize_dna(cds)
    return [normalized[index : index + 3] for index in range(0, len(normalized) - 2, 3)]


def translate_cds(cds: str) -> str:
    return "".join(CODON_TO_AA.get(codon, "X") for codon in split_codons(cds))


def internal_stop_codons(codons: list[str]) -> list[int]:
    return [
        index
        for index, codon in enumerate(codons, start=1)
        if CODON_TO_AA.get(codon) == "*" and index < len(codons)
    ]


def sequence_length_warnings(amino_acids: str | None) -> list[str]:
    if amino_acids is None:
        return []
    length = len(normalize_amino_acid_sequence(amino_acids))
    warnings = []
    if length < SHORT_AA_LENGTH:
        warnings.append(f"AA sequence is shorter than {SHORT_AA_LENGTH} residues.")
    if length > LONG_AA_LENGTH:
        warnings.append(
            f"AA sequence is longer than {LONG_AA_LENGTH} residues; inference and synthesis review may be slow."
        )
    return warnings


def check_translation(cds: str, amino_acids: str | None = None) -> TranslationCheck:
    normalized_cds = normalize_dna(cds)
    codons = split_codons(normalized_cds)
    translated = translate_cds(normalized_cds)
    expected = normalize_amino_acid_sequence(amino_acids) if amino_acids else None
    invalid_bases = sorted({base for base in normalized_cds if base not in DNA_BASES})
    return TranslationCheck(
        cds_length=len(normalized_cds),
        codon_count=len(codons),
        valid_dna=not invalid_bases,
        invalid_bases=invalid_bases,
        length_multiple_of_three=len(normalized_cds) % 3 == 0,
        translated_amino_acids=translated,
        translation_matches_input=(translated == expected) if expected is not None else None,
        internal_stop_codons=internal_stop_codons(codons),
        sequence_warnings=sequence_length_warnings(amino_acids),
    )
