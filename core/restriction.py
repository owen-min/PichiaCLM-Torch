from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .biology import normalize_dna


DEFAULT_RESTRICTION_ENZYMES = {
    "EcoRI": "GAATTC",
    "XhoI": "CTCGAG",
    "NotI": "GCGGCCGC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "NdeI": "CATATG",
    "NcoI": "CCATGG",
    "KpnI": "GGTACC",
    "XbaI": "TCTAGA",
    "SpeI": "ACTAGT",
}


@dataclass(frozen=True)
class RestrictionSite:
    name: str
    sequence: str
    start: int
    end: int


def parse_custom_sites(raw_sites: Iterable[str] | None) -> dict[str, str]:
    sites = {}
    if raw_sites is None:
        return sites
    for raw_site in raw_sites:
        item = raw_site.strip()
        if not item:
            continue
        if "=" in item:
            name, sequence = item.split("=", 1)
            name = name.strip()
        elif ":" in item:
            name, sequence = item.split(":", 1)
            name = name.strip()
        else:
            sequence = item
            name = f"custom_{normalize_dna(sequence)}"
        sequence = normalize_dna(sequence)
        if sequence:
            sites[name or f"custom_{sequence}"] = sequence
    return sites


def scan_restriction_sites(
    cds: str,
    include_defaults: bool = True,
    custom_sites: dict[str, str] | None = None,
) -> list[RestrictionSite]:
    normalized = normalize_dna(cds)
    site_map = dict(DEFAULT_RESTRICTION_ENZYMES) if include_defaults else {}
    if custom_sites:
        site_map.update({name: normalize_dna(sequence) for name, sequence in custom_sites.items() if sequence})

    hits = []
    for name, sequence in site_map.items():
        start = 0
        while True:
            index = normalized.find(sequence, start)
            if index == -1:
                break
            hits.append(RestrictionSite(name=name, sequence=sequence, start=index + 1, end=index + len(sequence)))
            start = index + 1
    return sorted(hits, key=lambda item: (item.start, item.name))
