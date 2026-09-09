from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .biology import AA_TO_CODONS, CODON_TO_AA, normalize_dna, split_codons, translate_cds
from .restriction import parse_custom_sites, scan_restriction_sites


@dataclass(frozen=True)
class CodonReplacement:
    codon_number: int
    start: int
    end: int
    amino_acid: str
    old_codon: str
    new_codon: str
    reason: str


@dataclass(frozen=True)
class PostprocessResult:
    original_cds: str
    optimized_cds: str
    amino_acids: str
    translation_preserved: bool
    replacements: list[CodonReplacement]
    fixed_issues: list[str]
    remaining_issues: list[str]


def conservative_postprocess(
    cds: str,
    amino_acids: str,
    reference_fractions: dict[str, float],
    forbidden_motifs: Iterable[str] | None = None,
    custom_restriction_sites: Iterable[str] | None = None,
    max_iterations: int = 200,
) -> PostprocessResult:
    original = normalize_dna(cds)
    target_aa = "".join(amino_acids.split()).upper()
    current = original
    replacements: list[CodonReplacement] = []
    fixed_issues: list[str] = []

    forbidden = [normalize_dna(item) for item in forbidden_motifs or [] if normalize_dna(item)]
    custom_sites = parse_custom_sites(custom_restriction_sites)

    for _ in range(max_iterations):
        issues = _current_issues(current, forbidden, custom_sites)
        if not issues:
            break
        changed = False
        for issue in issues:
            replacement = _replace_one_overlapping_codon(
                current,
                target_aa,
                reference_fractions,
                issue["start"],
                issue["end"],
                issue["label"],
            )
            if replacement is None:
                continue
            current = (
                current[: replacement.start - 1]
                + replacement.new_codon
                + current[replacement.end :]
            )
            replacements.append(replacement)
            fixed_issues.append(issue["label"])
            changed = True
            break
        if not changed:
            break

    remaining = [issue["label"] for issue in _current_issues(current, forbidden, custom_sites)]
    return PostprocessResult(
        original_cds=original,
        optimized_cds=current,
        amino_acids=target_aa,
        translation_preserved=translate_cds(current) == target_aa,
        replacements=replacements,
        fixed_issues=fixed_issues,
        remaining_issues=remaining,
    )


def _current_issues(cds: str, forbidden_motifs: list[str], custom_sites: dict[str, str]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for site in scan_restriction_sites(cds, include_defaults=True, custom_sites=custom_sites):
        issues.append({"start": site.start, "end": site.end, "label": f"restriction:{site.name}:{site.sequence}"})
    for motif in forbidden_motifs:
        start = 0
        while True:
            index = cds.find(motif, start)
            if index == -1:
                break
            issues.append({"start": index + 1, "end": index + len(motif), "label": f"motif:{motif}"})
            start = index + 1
    for issue in _homopolymer_issues(cds):
        issues.append(issue)
    for issue in _local_gc_issues(cds):
        issues.append(issue)
    for issue in _repeated_kmer_issues(cds):
        issues.append(issue)
    return sorted(issues, key=lambda item: (item["start"], item["end"]))


def _homopolymer_issues(cds: str, min_length: int = 6) -> list[dict[str, object]]:
    issues = []
    start = 0
    while start < len(cds):
        end = start + 1
        while end < len(cds) and cds[end] == cds[start]:
            end += 1
        if end - start >= min_length:
            issues.append({"start": start + 1, "end": end, "label": f"homopolymer:{cds[start]}:{end - start}"})
        start = end
    return issues


def _local_gc_issues(cds: str, window: int = 30, max_gc: float = 75.0) -> list[dict[str, object]]:
    issues = []
    if len(cds) < window:
        return issues
    for start in range(0, len(cds) - window + 1):
        segment = cds[start : start + window]
        gc_percent = (segment.count("G") + segment.count("C")) / window * 100
        if gc_percent > max_gc:
            issues.append({"start": start + 1, "end": start + window, "label": f"local_gc_high:{gc_percent:.2f}"})
    return issues


def _repeated_kmer_issues(cds: str, kmer_size: int = 12, min_count: int = 3) -> list[dict[str, object]]:
    positions: dict[str, list[int]] = {}
    if len(cds) < kmer_size:
        return []
    for start in range(0, len(cds) - kmer_size + 1):
        kmer = cds[start : start + kmer_size]
        positions.setdefault(kmer, []).append(start + 1)
    issues = []
    for kmer, starts in positions.items():
        if len(starts) >= min_count:
            for start in starts:
                issues.append({"start": start, "end": start + kmer_size - 1, "label": f"repeated_12mer:{kmer}"})
    return issues


def _replace_one_overlapping_codon(
    cds: str,
    target_aa: str,
    reference_fractions: dict[str, float],
    issue_start: int,
    issue_end: int,
    reason: str,
) -> CodonReplacement | None:
    codons = split_codons(cds)
    start_index = max(0, (int(issue_start) - 1) // 3)
    end_index = min(len(codons) - 1, (int(issue_end) - 1) // 3)

    for codon_index in range(start_index, end_index + 1):
        old_codon = codons[codon_index]
        aa = CODON_TO_AA.get(old_codon)
        if aa is None or aa == "*" or codon_index >= len(target_aa) or target_aa[codon_index] != aa:
            continue
        candidates = [
            codon for codon in AA_TO_CODONS.get(aa, [])
            if codon != old_codon
        ]
        if reason.startswith("local_gc_high"):
            candidates.sort(key=lambda codon: (codon.count("G") + codon.count("C"), -reference_fractions.get(codon, 0.0)))
        else:
            candidates.sort(key=lambda codon: reference_fractions.get(codon, 0.0), reverse=True)
        for new_codon in candidates:
            trial_codons = codons.copy()
            trial_codons[codon_index] = new_codon
            trial_cds = "".join(trial_codons)
            if translate_cds(trial_cds) != target_aa:
                continue
            return CodonReplacement(
                codon_number=codon_index + 1,
                start=codon_index * 3 + 1,
                end=codon_index * 3 + 3,
                amino_acid=aa,
                old_codon=old_codon,
                new_codon=new_codon,
                reason=reason,
            )
    return None
